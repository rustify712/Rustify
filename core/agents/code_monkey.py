import asyncio
import json
import os
from collections import defaultdict
from typing import Dict, List, Literal, Optional
import re
from pydantic import BaseModel, Field
import numpy as np

from core.agents.base import AgentResponse, AgentResponseStatus, BaseAgent
from core.agents.transpile_memory import TranspileMemory
from core.utils.prompt_loader import PromptLoader
from core.schema.response import CodeMonkeyResponseType
from core.schema.translation import TranslationTask, TranslationTaskStatus, TranslationUnitNode
from core.state.state_manager import ModuleTranslation, StateManager
from core.tools.file_tools import read_file_tool
from core.utils.file_utils import add_line_numbers
from core.utils.rust_utils import cargo_check
from core.utils.tree_search import GreedySearchStrategy, TreeSearchNode, TreeSearch
from core.utils.optimizer import TemperatureOptimizer
from core.utils.patch import extract_code_block_change_info, apply_changes
from core.constants.experiences import FIXING_EXPERIENCES


class TranslateResponse(BaseModel):
    filepath: str = Field(
        description="转译后的 Rust 代码所在文件相对路径, 应该与原 C/C++ 文件名对应遵守 Rust 文件命名规范。")
    type: Literal["code", "comment"] = Field(
        description="转译结果的类型，当存在 Rust 代码时为 code，当无 Rust 代码时为 comment。")
    summary: str = Field(
        description="转译结果摘要，通常为函数（方法）或结构体的名称或注释摘要，不应该超过20个字。")
    text: str = Field(description="转译后的 Rust 代码，若无 Rust 代码为空字符串即可。")
    explanation: Optional[str] = Field(description="转译说明文本和描述。")


class ReviewTranslationResponse(BaseModel):
    suggestions: List[str] = Field(description="每个转译结果的详细评审意见")
    best_one: int = Field(description="最佳转译结果的序号")


class ReviewSummaryResponse(BaseModel):
    issues: List[str] = Field(description="详细问题列表")
    suggestions: List[str] = Field(description="详细建议列表")
    passed: bool = Field(description="是否满足任务要求的情况进行打分，True 为满足，False 为不满足。")


class ExperienceQuery(BaseModel):
    queries: List[str] = Field(description="查询的问题列表")


def compute_translation_score(errors: list[dict]) -> int:
    """计算转译错误分数, 根据错误类型和错误数量"""
    import math
    error_type_count_map = defaultdict(int)
    for error in errors:
        if error["code"] is None or error["code"]["code"] is None:
            error_type_count_map["None"] += 1
        else:
            error_type_count_map[error["code"]["code"]] += 1
    error_score = 0
    for error_type, error_count in error_type_count_map.items():
        error_score += 1 + math.log(error_count)
    if error_score > 100:
        error_score = 100
    return 100 - error_score


def write_code(filepath: str, code: str, append: bool = False):
    """写入代码"""
    if append:
        with open(filepath, "r", encoding="utf-8") as f:
            file_content = f.read()
        with open(filepath, "a", encoding="utf-8") as f:
            if file_content.strip() != "":
                f.write("\n\n" + code)
            else:
                f.write(code)
    else:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)


class CodeMonkey(BaseAgent):
    """转译智能体

    TODO: 添加产生多个转译结果, 取错误最少的结果作为最终结果.
    TODO: 将错误修复部分抽离, 单独作为一个智能体.

    ```mermaid
    graph TD;

    ```
    """
    ROLE = "code_monkey"
    DESCRIPTION = "A powerful and efficient AI assistant that completes tasks accurately and resourcefully, providing coding support and handling routine tasks with precision."

    task_temperature_optimizer_map: Dict[str, TemperatureOptimizer] = {
        "translate": TemperatureOptimizer(
            initial_temp=0.2,
            min_temp=0.0,
            max_temp=1.3,
            sigma=0.2,
        ),
        "fixing": TemperatureOptimizer(
            initial_temp=0.4,
            min_temp=0.0,
            max_temp=1.3,
            sigma=0.2,
        )
    }
    """任务温度优化器映射, 为每一个任务维护一个温度优化器。"""

    def __init__(
            self,
            llm_config: dict,
            state_manager: "StateManager",
            **kwargs
    ):
        super().__init__(llm_config, **kwargs)
        self.state_manager = state_manager
        self.module_name: Optional[str] = None
        self.module_translation: Optional[ModuleTranslation] = None
        self.translation_task: Optional[TranslationTask] = None
        # 多采样
        self.translation_samples_num = 5
        self.fixing_samples_num = 5
        # 当前智能体的消息记录
        # 推理消息
        self.reasoning_messages = []
        # 转译消息
        self.translation_messages = []
        # 修复消息
        self.fixing_messages = []

        # 转译候选项
        """
        {
            "messages": [dict],  # 转译过程中的对话消息
            "translation": TranslateResponse,  # 转译结果
            "temperature": float,  # 转译时的温度
            "check_result": dict,  # 转译后的 Rust 代码编译结果
            "has_tried": bool,  # 是否已经尝试过
        }
        """
        self.translation_candidates: list[dict] = []
        # 当前采纳的转译结果
        self.translation_completion: Optional[dict] = None
        # 记忆模块
        self.long_memory = TranspileMemory(llm_config, state_manager)
        self.translation_experience = []

        # TODO：暂时不使用工具，对于 DeepSeek 模型，不能保证该模型是否能够在该使用工具的场景下正确使用工具。
        # self.register_tool(modify_file_tool)
        # self.register_tool(read_file_tool)
        # self.register_tool(file_insert_content_tool)

    def run(self, pre_response: AgentResponse) -> AgentResponse:
        if pre_response.status == AgentResponseStatus.DONE:
            if pre_response.type == CodeMonkeyResponseType.PREPARE_TRANSLATION_TASK:
                # 转译预处理后，开始进行经验查询
                # return self.query_translation_experience_step(pre_response)
                return asyncio.run(self.translate(pre_response))
            elif pre_response.type == CodeMonkeyResponseType.TRANSLATION_EXPERIENCE_QUERY:
                # 经验查询完成后，开始进行转译
                return asyncio.run(self.translate(pre_response))
            elif pre_response.type == CodeMonkeyResponseType.TRANSLATION_COMPLETION:
                # 转译完成后，评估转译结果
                return self.evaluate_translation(pre_response)
            # elif pre_response.type in [CodeMonkeyResponseType.WRITE_CODE, CodeMonkeyResponseType.FIX_CODE]:
            #     # 写入代码完成，检查代码
            #     return self.check_code(pre_response)
            elif pre_response.type == CodeMonkeyResponseType.CHECK_CODE_FAILED:
                # 代码检查未通过，根据错误进行反思
                return AgentResponse.done(self, CodeMonkeyResponseType.TRANSLATION_TASK_DONE)
                return self.fixing(pre_response)
            elif pre_response.type == CodeMonkeyResponseType.FIX_FAILED:
                # 错误修复超出限制
                return self.mark_as_todo(pre_response)
        elif pre_response.type == AgentResponseStatus.ERROR:
            return AgentResponse.error(self, CodeMonkeyResponseType.TRANSLATION_TASK_DONE, error=pre_response.data)

    def start(self, module_name: str, translation_task: TranslationTask) -> AgentResponse:
        self.logger.info(f"Code Monkey [{self.name}] started.")
        agent_response = self.prepare_translation_tasks_step(module_name, translation_task)
        while True:
            agent_response = self.run(agent_response)
            if agent_response is None:
                return AgentResponse.error(self, CodeMonkeyResponseType.TRANSLATION_TASK_DONE, error={
                    "message": "No response from the agent."
                })
            if agent_response.type == CodeMonkeyResponseType.TRANSLATION_TASK_DONE:
                # 所有转译任务完成
                return agent_response

    def prepare_translation_tasks_step(self, module_name: str, translation_task: TranslationTask) -> AgentResponse:
        self.module_name = module_name
        self.module_translation = self.state_manager.translator.module_translations[module_name]
        self.translation_task = translation_task
        self.logger.info(f"Preparing translation tasks: [{translation_task.source.id}] in module [{module_name}]")
        return AgentResponse.done(self, CodeMonkeyResponseType.PREPARE_TRANSLATION_TASK)

    def query_translation_experience_step(self, agent_response: AgentResponse) -> AgentResponse:
        """查询转译经验"""
        file_summary = self.state_manager.source_project.file_summaries.get(
            self.translation_task.source.nodes[0].filepath, "")
        experience_queries_prompt = PromptLoader.get_prompt(
            f"{self.ROLE}/experience_queries.prompt",
            current_translation_task=self.translation_task,
            file_summary=file_summary,
        )
        self.logger.debug(f"Experience queries prompt: \n{experience_queries_prompt}")
        experience_queries_response = self.call_llm(messages=[{"role": "user", "content": experience_queries_prompt}],
                                                    json_format=ExperienceQuery)
        if experience_queries_response.format_object:
            queries = experience_queries_response.format_object.queries
        else:
            experience_queries_response_content = experience_queries_response.choices[0].message.content
            query_match = re.search(r"\[[\s\S*?]\]", experience_queries_response_content)
            if query_match:
                queries = query_match.group().strip("[]").split(",")
            else:
                queries = []
        self.logger.debug(f"Experience queries: {queries}")
        if not self.long_memory:
            self.translation_experience = []
        self.translation_experience = self.long_memory.query_translation_memory(queries)
        return AgentResponse.done(self, CodeMonkeyResponseType.TRANSLATION_EXPERIENCE_QUERY)

    def translate_with_call_llm_task(
            self,
            translate_messages: list[dict],
            temperature: float,
            task_id: str
    ) -> dict | None:
        translate_messages = translate_messages.copy()
        try_count = 0
        while try_count < 3:
            response = self.call_llm(messages=translate_messages, temperature=temperature)
            translate_message_content = response.choices[0].message.content
            self.logger.debug(f"[Task {task_id}] Translate response content: \n{translate_message_content}")
            translate_messages.append({"role": "assistant", "content": translate_message_content})
            # 提取所有的 JSON
            json_blocks = re.findall(r"```json\s*(.*?)\s*```", translate_message_content, re.DOTALL)
            if len(json_blocks) == 0:
                # self.logger.error("No JSON block found in the response.")
                translate_messages.append({"role": "user", "content": "No JSON block found in the response."})
            else:
                # TODO: 更加复杂的校验规则，这里仅仅简单的获取最后一个 JSON 代码块
                json_block = json_blocks[-1]
                try:
                    translate_response_obj = TranslateResponse.model_validate_json(json_block)
                    self.logger.debug(
                        f"[Task {task_id}] Translate response code: \n{translate_response_obj.explanation}\n{translate_response_obj.text}")
                    return {
                        "messages": translate_messages,
                        "translation": translate_response_obj
                    }
                except Exception as e:
                    # self.logger.error(f"Error occurred when parsing JSON: {e}")
                    translate_messages.append({"role": "user", "content": "Error occurred when parsing JSON."})
            try_count += 1
        return None

    async def translate(self, pre_response: AgentResponse) -> AgentResponse:
        """转译代码
        根据不同的温度生成多个转译候选
        """
        if self.translation_task.status == TranslationTaskStatus.DONE:
            # 当前转译任务已完成
            self.logger.info(
                f"Translation task done: {self.translation_task.source.name}({self.translation_task.source.id})")
            return AgentResponse.done(self, CodeMonkeyResponseType.TRANSLATION_TASK_DONE)
        self.logger.info(
            f"Translation task started: {self.translation_task.source.name}({self.translation_task.source.id})")
        # 1. 将当前转译任务状态标记为运行中
        self.state_manager.mark_translation_task_as(self.module_name, self.translation_task.source.id,
                                                    TranslationTaskStatus.RUNNING)
        # 2. 构建转译任务的提示词
        # 2.1 获取相关的转译任务，目前传递相关文件的内容，故暂时不考虑相关转译任务
        # 2.2 获取相关的 Rust 文件
        related_project_files = self.state_manager.target_project.list_files(
            show_content=True,
            # 忽略不在 self.module_translation.related_rust_files 中的文件
            ignore_func=lambda filepath: os.path.isfile(os.path.join(self.state_manager.target_project.path,
                                                                     filepath)) and filepath not in self.module_translation.related_rust_files
        )
        # 2.3 获取相关的节点类型
        related_node_types = list(set([
            node.type
            for node in self.translation_task.source.nodes
        ]))
        # 2.4 获取相关的节点类型，目前通过 reasoning 进行推理，暂时不考虑相关节点类型
        # 2.5 获取当前项目结构
        project_structure = self.state_manager.target_project.pretty_structure(
            # 忽略不在 related_project_files + ["Cargo.toml", "src/main.rs", "src/lib.rs"] 中的文件
            lambda filepath: os.path.isfile(
                os.path.join(self.state_manager.target_project.path, filepath)) and filepath not in [
                                 project_file.path for project_file in related_project_files
                             ] + ["Cargo.toml", "src/main.rs", "src/lib.rs"]
        )
        # 2.6 获取相关的转译任务
        related_translation_tasks = [
            self.module_translation.get_translation_task_by_task_id(task_id)
            for task_id in self.translation_task.prerequisites
        ]
        # 找到当然任务的索引
        current_task_index = self.module_translation.translation_tasks.index(self.translation_task)
        # 当前任务的前10和后20个任务
        start_task_index = max(0, current_task_index - 10)
        end_task_index = min(len(self.module_translation.translation_tasks), current_task_index + 20)
        translation_tasks = self.module_translation.translation_tasks[start_task_index:end_task_index]
        # 2.6 生成推理提示词
        translate_prompt = PromptLoader.get_prompt(
            f"{self.ROLE}/translate.prompt",
            # 全部的转译任务，用于帮助模型选择合适的类型等
            # translation_tasks=self.module_translation.translation_tasks,
            # 部分转译任务
            translation_tasks=translation_tasks,
            # 相关转译任务
            related_translation_tasks=related_translation_tasks,
            # 当前转译任务
            current_translation_task=self.translation_task,
            # 相关文件列表及内容
            project_files=related_project_files,
            show_file_content=True,
            # Rust 项目结构及描述
            project_description=self.state_manager.target_project.description,
            project_structure=project_structure,
            # 相关节点类型
            node_types=related_node_types,
            json_schema=TranslateResponse.model_json_schema()
        )
        self.logger.debug(f"Translate prompt: \n{translate_prompt}")
        # 3. 根据温度生成多个转译结果
        # 3.1 生成多个温度
        translate_temperature_optimizer = self.task_temperature_optimizer_map["translate"]
        temperatures = translate_temperature_optimizer.do_sample(self.translation_samples_num)
        # 3.2 并发生成多个转译结果
        translate_coros = [
            asyncio.to_thread(self.translate_with_call_llm_task, [{
                "role": "user",
                "content": translate_prompt
            }], temperature, str(index))
            for index, temperature in enumerate(temperatures)
        ]
        # 3.3 等待所有的结果
        translate_coro_results = await asyncio.gather(*translate_coros)
        # 3.4 记录所有的转译结果
        self.translation_candidates = []
        for temperature, translate_coro_result in zip(temperatures, translate_coro_results):
            if translate_coro_result is None:
                continue
            self.translation_candidates.append({
                "messages": translate_coro_result["messages"],
                "translation": translate_coro_result["translation"],
                "temperature": temperature,
            })
        return AgentResponse.done(self, CodeMonkeyResponseType.TRANSLATION_COMPLETION)

    def evaluate_translation(self, pre_response: AgentResponse) -> AgentResponse:
        """评估候选中的转译结果，选择最佳的转译结果"""
        untried_candidates = [
            candidate
            for candidate in self.translation_candidates
            if not candidate.get("has_tried", False)
        ]
        if len(untried_candidates) == 0:
            # 所有候选结果均已尝试，但未成功
            return AgentResponse.done(self, CodeMonkeyResponseType.TRANSLATION_COMPLETION_FAILED)
        # 为每一个候选结果评估分数
        for candidate in untried_candidates:
            translation = candidate["translation"]
            if translation.type == "comment":
                # 无 Rust 代码，直接跳过
                candidate["score"] = 100
                continue
            else:
                # 评估候选结果，计算分数, 分数主要取决于编译错误的类型数量，以及错误数量
                rust_node_code = translation.text
                rust_file = translation.filepath
                rust_filepath = os.path.join(self.state_manager.target_project.path, rust_file)
                if rust_file not in self.module_translation.related_rust_files:
                    # 该 Rust 文件不在相关 Rust 文件列表中，添加到相关 Rust 文件列表中
                    self.state_manager.add_module_rust_files(
                        self.module_name, [rust_file]
                    )
                # 锁定该文件，临时记录 Rust 文件内容，以便后续恢复
                with self.state_manager.file_lock_manager.file_lock(rust_filepath):
                    if not os.path.exists(rust_filepath):
                        os.makedirs(os.path.dirname(rust_filepath), exist_ok=True)
                        with open(rust_filepath, "w", encoding="utf-8") as f:
                            f.write("")
                    # 临时记录 Rust 文件内容
                    with open(rust_filepath, "r", encoding="utf-8") as f:
                        temp_rust_file_content = f.read()
                    # 写入 Rust 代码
                    write_code(rust_filepath, rust_node_code, append=True)
                    # 读取 Rust 文件内容
                    with open(rust_filepath, "r", encoding="utf-8") as f:
                        candidate["code"] = f.read()
                    # 检查 Rust 代码，忽略 E0601 错误：main 函数未找到
                    cargo_check_result = cargo_check(self.state_manager.target_project.path, [rust_filepath],
                                                     ignore_codes=["E0601"])
                    candidate["check_result"] = cargo_check_result
                    # 恢复 Rust 文件内容
                    with open(rust_filepath, "w", encoding="utf-8") as f:
                        f.write(temp_rust_file_content)
                if cargo_check_result["success"]:
                    candidate["score"] = 100
                else:
                    candidate["score"] = compute_translation_score(cargo_check_result["errors"])
        # 采取贪心算法，率先尝试最高分数的结果
        # 最大分数
        max_score = max([candidate["score"] for candidate in untried_candidates])
        # 检查最大分数的数量
        max_score_count = sum(
            [1 for candidate in untried_candidates if candidate["score"] == max_score])
        best_candidate = None
        if max_score_count <= 1:
            # 最大分数数量小于等于1，直接选择分值最高的结果
            best_candidate = max(untried_candidates, key=lambda x: x["score"])
        else:
            # 最大分数数量大于1，通过 review 进行选择
            max_score_candidates = [candidate for candidate in untried_candidates if
                                    candidate["score"] == max_score]
            best_candidate = max_score_candidates[0]
            # rust_codes = [
            #     candidate["translation"].explanation + "\n" + candidate["translation"].text
            #     for candidate in max_score_candidates
            # ]
            # # 获取相关文件
            # related_project_files = self.state_manager.target_project.list_files(
            #     show_content=True,
            #     # 忽略不在 self.module_translation.related_rust_files 中的文件
            #     ignore_func=lambda filepath: os.path.isfile(os.path.join(self.state_manager.target_project.path,
            #                                                              filepath)) and filepath not in self.module_translation.related_rust_files
            # )
            # # 获取当前项目结构
            # project_structure = self.state_manager.target_project.pretty_structure(
            #     # 忽略不在 related_project_files + ["Cargo.toml", "src/main.rs", "src/lib.rs"] 中的文件
            #     lambda filepath: os.path.isfile(
            #         os.path.join(self.state_manager.target_project.path, filepath)) and filepath not in [
            #                          project_file.path for project_file in related_project_files
            #                      ] + ["Cargo.toml", "src/main.rs", "src/lib.rs"]
            # )
            # # 获取相关节点类型
            # related_node_types = list(set([
            #     node.type
            #     for node in self.translation_task.source.nodes
            # ]))
            # # 获取相关转译任务
            # related_translation_tasks = [
            #     self.module_translation.get_translation_task_by_task_id(task_id)
            #     for task_id in self.translation_task.prerequisites
            # ]
            # review_translation_prompt = PromptLoader.get_prompt(
            #     f"{self.ROLE}/review_translation.prompt",
            #     translation_tasks=self.module_translation.translation_tasks,
            #     current_translation_task=self.translation_task,
            #     related_translation_tasks=related_translation_tasks,
            #     project_description=self.state_manager.target_project.description,
            #     project_structure=project_structure,
            #     project_files=related_project_files,
            #     show_file_content=True,
            #     node_types=related_node_types,
            #     rust_codes=rust_codes
            # )
            # self.logger.debug(f"Review translation prompt: \n{review_translation_prompt}")
            # try_count = 0
            # while try_count < 3:
            #     review_translation_messages = [
            #         {"role": "user", "content": review_translation_prompt}
            #     ]
            #     review_translation_response = self.call_llm(messages=review_translation_messages,
            #                                                 json_format=ReviewTranslationResponse)
            #     review_translation_messages.append(
            #         {"role": "assistant", "content": review_translation_response.choices[0].message.content})
            #     review_translation_response_obj = review_translation_response.format_object
            #     if review_translation_response_obj is None:
            #         # 未能解析 ReviewTranslationResponse
            #         review_translation_messages.append({"role": "user", "content": "Error occurred when parsing JSON."})
            #         try_count += 1
            #         continue
            #     else:
            #         self.logger.debug(
            #             f"Review translation response: \n{json.dumps(review_translation_response_obj.dict(), indent=4, ensure_ascii=False)}")
            #         best_candidate = max_score_candidates[review_translation_response_obj.best_one - 1]
            #         break
        if best_candidate is None:
            best_candidate = untried_candidates[0]
        # 更新当前采纳的转译结果
        self.translation_completion = best_candidate
        self.translation_messages = best_candidate["messages"]
        best_translation = best_candidate["translation"]
        best_translation_temperature = best_candidate["temperature"]
        best_translation_score = best_candidate["score"]
        best_translation_code_snippet = best_translation.text
        self.logger.debug(
            f"Best candidate: \nTemperature: {best_translation_temperature}, Score: {best_translation_score}\nCode Snippet: \n{best_translation_code_snippet}")
        # 更新转译任务最佳温度
        temperature_optimizer = self.task_temperature_optimizer_map["translate"]
        temperature_optimizer.update(best_translation_temperature)
        self.translation_task.target = TranslationUnitNode(
            filepath=best_translation.filepath,
            id=self.translation_task.source.id,
            name=best_translation.summary,
            type=best_translation.type,
            text=best_translation.text,
            description=best_translation.explanation,
        )
        # if best_translation_score == 100:
        if True:
            # 无编译错误，无需进行后续的修错
            if best_translation.type == "code":
                best_translation.filepath = "src/libtree.rs"
                best_translation_code_filepath = os.path.join(self.state_manager.target_project.path,
                                                              best_translation.filepath)
                with self.state_manager.file_lock_manager.file_lock(best_translation_code_filepath):
                    write_code(best_translation_code_filepath, best_translation.text, append=True)
                    # 添加到版本控制
                    self.state_manager.target_vcs.add([best_translation_code_filepath])
                    self.state_manager.target_vcs.commit(
                        f"translation task {self.translation_task.source.name}({self.translation_task.source.id}) done.")
            # self.translation_task.target = TranslationUnitNode(
            #     filepath=best_translation.filepath,
            #     id=self.translation_task.source.id,
            #     name=best_translation.summary,
            #     type=best_translation.type,
            #     text=best_translation.text,
            #     description=best_translation.explanation,
            # )
            # self.state_manager.mark_translation_task_as(
            #     self.module_name, self.translation_task.source.id, TranslationTaskStatus.DONE
            # )
            self.logger.info(
                f"Translation task {self.translation_task.source.name}({self.translation_task.source.id}) done.")
            # self.long_memory.store_translation_memory(
            #     self.module_translation, self.translation_task, self.translation_messages
            # )
            return AgentResponse.done(self, CodeMonkeyResponseType.TRANSLATION_TASK_DONE)
        else:
            # 有编译错误，需要进行后续的修错
            self.logger.debug(f"Translation task occurred errors: \n" + "\n".join([
                error["rendered"]
                for error in best_candidate["check_result"]["errors"]
            ]))

            return AgentResponse.done(self, CodeMonkeyResponseType.CHECK_CODE_FAILED)

    def fixing_with_call_llm_task(
            self,
            rust_code: str,
            fix_errors_messages: list[dict],
            temperature: float,
            task_id: str
    ) -> dict | None:

        fix_errors_messages = fix_errors_messages.copy()
        fix_errors_prompt_message = fix_errors_messages[-1]
        try_count = 0
        fix_errors_message_content = ""
        while try_count < 3:
            response = self.call_llm(messages=fix_errors_messages, temperature=temperature)
            fix_errors_message_content += response.choices[0].message.content
            fix_errors_messages.append({"role": "assistant", "content": fix_errors_message_content})
            if response.choices[0].finish_reason == "length":
                # 问答对话长度过长，需要重新继续
                fix_errors_messages.append({"role": "user", "content": "continue"})
                continue
            self.logger.debug(f"[Task {task_id}] fixing response content: \n{fix_errors_message_content}")

            changes_info = extract_code_block_change_info(fix_errors_message_content)
            if len(changes_info) == 0:
                self.logger.error("No Rust code block found in the response.")
                fix_errors_messages.append({"role": "user", "content": "No Rust code block found in the response."})
                try_count += 1
                continue
            else:
                # TODO: 更加复杂的校验规则或利用工具，这里仅仅简单的获取最后一个 Rust 代码块
                filepath, changes = list(changes_info.items())[0]
                rust_code = apply_changes(rust_code, changes)
                return {
                    "messages": [
                        fix_errors_prompt_message,
                        {"role": "assistant", "content": fix_errors_message_content}
                    ],
                    "code": rust_code
                }
        return None

    def fixing(self, pre_response: AgentResponse) -> AgentResponse:
        """修复编译错误"""
        fixing_temperature_optimizer = self.task_temperature_optimizer_map["fixing"]

        filepath = os.path.join(
            self.state_manager.target_project.path,
            self.translation_completion["translation"].filepath
        )
        # 修复错误需要保证当前文件没有任何修改，对文件加锁
        self.state_manager.file_lock_manager.acquire_file_lock(filepath)
        # 读取文件内容，并在文件末尾添加转译后的 Rust 代码
        with open(filepath, "r", encoding="utf-8") as f:
            init_rust_code = f.read()
        init_rust_code += "\n\n" + self.translation_completion["translation"].text

        # 定义并发产生多个修复方案的函数
        def _fixing_expansion_func(node: TreeSearchNode) -> list[dict]:
            """根据错误, 生成多个修复结果"""

            async def async_fixing_expansion_func(node: TreeSearchNode):
                errors = node.data.get("errors")
                experiences = set()
                error_codes = set([
                    error["code"]["code"]
                    for error in errors
                    if error["code"] and error["code"]["code"]
                ])
                if error_codes:
                    for codes, experience in FIXING_EXPERIENCES.items():
                        if error_codes.intersection(codes):
                            experiences.add(experience)

                if node.data.get("try_count", 0) == 0:
                    # 第一次修复时需要提供全部的代码内容
                    code = node.data.get("code")
                    fix_errors_prompt = PromptLoader.get_prompt(
                        f"{self.ROLE}/fixing.prompt",
                        code=add_line_numbers(code),
                        errors=errors,
                        experiences=experiences
                    )
                else:
                    # 后续修复时只需要提供错误信息，无需提供全部的代码内容
                    fix_errors_prompt = PromptLoader.get_prompt(
                        f"{self.ROLE}/fixing.prompt",
                        errors=errors
                    )
                self.logger.debug(f"Fixing prompt: \n{fix_errors_prompt}")
                history_messages = self.reasoning_messages + self.translation_messages + self.fixing_messages
                fix_errors_messages = history_messages + [
                    {"role": "user", "content": fix_errors_prompt}
                ]
                temperatures = fixing_temperature_optimizer.do_sample(self.fixing_samples_num)
                fix_errors_coros = [
                    asyncio.to_thread(self.fixing_with_call_llm_task, node.data.get("code"), fix_errors_messages, temperature, str(index))
                    for index, temperature in enumerate(temperatures)
                ]
                fix_errors_coro_results = await asyncio.gather(*fix_errors_coros)
                fix_errors_results = []
                for temperature, coro_result in zip(temperatures, fix_errors_coro_results):
                    if coro_result is not None:
                        fix_errors_results.append({
                            "messages": coro_result["messages"],
                            "code": coro_result["code"],
                            "temperature": temperature,
                            "try_count": node.data.get("try_count") + 1
                        })
                self.logger.debug(f"[Fix Errors Expansion {node.data.get('try_count')}]: \n" + "\n\n".join([
                    f"Temperature: {fix_errors_result['temperature']}\nMessage Content: {fix_errors_result['messages'][-1]['content']}\nRust Code: \n{fix_errors_result['code']}"
                    for fix_errors_result in fix_errors_results
                ]))
                return fix_errors_results

            return asyncio.run(async_fixing_expansion_func(node))

        # 定义评估修复方案的函数
        def _fixing_compare_func(node1: TreeSearchNode, node2: TreeSearchNode) -> bool:
            """根据错误, 选择最优的修复结果
            选择错误最少的结果
            """
            if len(node1.data.get("errors")) > len(node2.data.get("errors")):
                return True
            return False

        # 定义模拟修复方案的函数
        def _fixing_simulation_func(node: TreeSearchNode) -> float:
            """评估当前代码的错误
            """
            with open(filepath, "r", encoding="utf-8") as f:
                temp_file_content = f.read()
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(node.data.get("code"))
            cargo_check_result = cargo_check(self.state_manager.target_project.path, [filepath], ignore_codes=["E0601"])
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(temp_file_content)
            node.data["errors"] = cargo_check_result["errors"]
            self.logger.debug(f"[Fix Errors Simulation] Rust Code\n{node.data.get('code')}\nErrors: " + "\n".join([
                error["rendered"]
                for error in node.data["errors"]
            ]))
            return len(cargo_check_result["errors"])

        # 定义选择修复方案的函数
        def _fixing_selection_func(nodes: list[TreeSearchNode]) -> TreeSearchNode:
            """选择最优的修复结果
            选择错误最少的结果
            """
            best_node = min(nodes, key=lambda x: len(x.data.get("errors")))
            best_node_messages = best_node.data.get("messages")
            fixing_temperature_optimizer.update(best_node.data.get("temperature"))
            self.logger.debug(
                f"[Fix Errors Selection]: \nmessage content: {best_node_messages[-1]['content']}\nBest Node Rust Code: {best_node.data.get('code')}\nErrors: " + "\n".join(
                    [
                        error["rendered"]
                        for error in best_node.data["errors"]
                    ]))
            # 记录当前最佳修复结果的对话消息
            if len(self.fixing_messages) == 0:
                self.fixing_messages = best_node_messages
            else:
                self.fixing_messages = self.fixing_messages[:-2] + best_node_messages
            return best_node

        # 定义目标条件的函数
        def _fixing_target_func(node: TreeSearchNode) -> bool:
            """判断是否达到目标"""
            if len(node.data.get("errors")) == 0:
                self.logger.debug(
                    f"[Fix Errors Target] \nmessage content: {node.data.get('messages')[-1]['content']}\nRust Code\n{node.data.get('code')}\nErrors: " + "\n".join(
                        [
                            error["rendered"]
                            for error in node.data["errors"]
                        ]))
                return True
            return False

        strategy = GreedySearchStrategy(
            expansion_func=_fixing_expansion_func,
            simulation_func=_fixing_simulation_func,
            selection_func=_fixing_selection_func,
            compare_func=_fixing_compare_func,
            max_depth=8,  # 最大尝试次数
            max_expansion=100  # 最大扩展次数
        )

        tree_search = TreeSearch(
            root_state={
                "try_count": 0,
                "code": init_rust_code,
                "messages": self.translation_messages
            },
            search_strategy=strategy
        )
        best_node = tree_search.search(_fixing_target_func)
        self.logger.debug(f"{strategy.strategy_name} Tree Search: \n{tree_search.print()}")

        if len(best_node.data.get("errors")) == 0:
            # 修复成功
            code = best_node.data.get("code")
            write_code(filepath, code, append=False)
            self.state_manager.target_vcs.add([filepath])
            self.state_manager.target_vcs.commit(
                f"translation task {self.translation_task.source.name}({self.translation_task.source.id}) done by fixing.")
            self.state_manager.file_lock_manager.release_file_lock(filepath)
            # self.state_manager.mark_translation_task_as(
            #     self.module_name, self.translation_task.source.id, TranslationTaskStatus.DONE
            # )
            self.logger.info(
                f"Translation task {self.translation_task.source.name}({self.translation_task.source.id}) done.")
            # self.long_memory.store_translation_memory(
            #     self.module_translation, self.translation_task, self.translation_messages + self.fixing_messages
            # )
            return AgentResponse.done(self, CodeMonkeyResponseType.TRANSLATION_TASK_DONE)
        else:
            self.state_manager.file_lock_manager.release_file_lock(filepath)
            # 修复失败
            self.logger.info(
                "Translation task {self.translation_task.source.name}({self.translation_task.source.id}) fixing failed.")
            return AgentResponse.done(self, CodeMonkeyResponseType.FIX_FAILED)

    def mark_as_todo(self, pre_response: AgentResponse) -> AgentResponse:
        """将当前任务标记为待办"""
        # 进行转译
        self.logger.info(
            f"Mark todo translation task: {self.translation_task.source.name}({self.translation_task.source.id})")
        mark_as_todo_prompt = PromptLoader.get_prompt(
            f"{self.ROLE}/mark_as_todo.prompt",
            current_task=self.translation_task,
        )
        self.logger.debug(f"Mark as todo prompt: \n{mark_as_todo_prompt}")
        mark_as_todo_messages = [{
            "role": "user",
            "content": mark_as_todo_prompt
        }]
        try_count = 0
        while try_count < 3:
            mark_as_todo_response = self.call_llm(mark_as_todo_messages, json_format=TranslateResponse)
            mark_as_todo_messages.append(
                {"role": "assistant", "content": mark_as_todo_response.choices[0].message.content})
            translate_response_obj = mark_as_todo_response.format_object
            if translate_response_obj is None:
                mark_as_todo_messages.append({
                    "role": "user",
                    "content": "Failed to parse JSON. Retry."
                })
            else:
                self.translation_task.target.text = translate_response_obj.text
                self.translation_task.target.description = translate_response_obj.explanation
                self.logger.debug(f"Mark as todo response description: \n{self.translation_task.target.description}")
                self.logger.debug(f"Mark as todo response text: \n{self.translation_task.target.text}")
                return AgentResponse.done(self, CodeMonkeyResponseType.MARK_TODO_DONE)
            try_count += 1
        return AgentResponse.error(self, CodeMonkeyResponseType.MARK_TODO_DONE, error={
            "message": "Failed to parse JSON as TranslateResponse.",
        })
