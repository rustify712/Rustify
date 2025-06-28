import asyncio
import os
from collections import defaultdict
from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from core.agents.reasoner import Reasoner
from core.agents.base import AgentResponse, AgentResponseStatus, BaseAgent
from core.llms.base_client import Tool, parse_json_format
from core.translator import Translator
from core.utils.patch import apply_changes, extract_code_block_change_info
from core.utils.prompt_loader import PromptLoader
from core.schema.response import CodeMonkeyResponseType
from core.schema.translation import TranslationTask, TranslationTaskStatus, TranslationTaskTarget, TranslationUnitNode
from core.state.state_manager import ModuleTranslation
from core.utils.optimizer import TemperatureOptimizer
from core.utils.rust_utils import cargo_check


class TranslateResponse(BaseModel):
    filepath: str = Field(
        description="转译后的 Rust 代码所在文件相对路径, 应该与原 C/C++ 文件名对应遵守 Rust 文件命名规范。")
    type: Literal["code", "comment"] = Field(
        description="转译结果的类型，当存在 Rust 代码时为 code，当无 Rust 代码时为 comment。")
    name: str = Field(description="转译后的 Rust 代码摘要，通常为函数（方法）或结构体的名称或注释摘要，不应该超过20个字。")
    text: str = Field(description="转译后的 Rust 代码，若无 Rust 代码为空字符串即可。")
    explanation: Optional[str] = Field(description="转译采取的策略或转译过程的解释，可为空字符串。")


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
            translator: Translator,
            **kwargs
    ):
        super().__init__(translator.llm_config, **kwargs)
        self.translator = translator

        self.module_translation_id: Optional[str] = None
        self.translation_task_id: Optional[str] = None
        # 推理智能体
        self.reasoner = Reasoner(self.translator.reasoner_config, logger=self.logger)
        # 记忆
        # self.memory = TranspileMemory(translator, logger=self.logger)
        # 当前任务对话记录
        self.translate_messages = []
        # 当前代码的编译错误
        self.compile_errors = []
        # 转译候选项
        """
        {
            "messages": [dict],  # 转译过程中的对话消息
            "translation": TranslateResponse,
            "temperature": float,
            "compile_result": dict,
            "has_tried": bool,
            "code": str
        }
        """
        self.translation_candidate_list = []
        # 当前采纳的转译结果
        self.translation_candidate = None
        self.translation_context: Optional[str] = None

        # # 多采样
        # self.translation_samples_num = 5
        # self.fixing_samples_num = 5
        # # 当前智能体的消息记录
        # # 推理消息
        # self.reasoning_messages = []
        # # 转译消息
        # self.translation_messages = []
        # # 修复消息
        # self.fixing_messages = []
        #
        # # 转译候选项
        # """
        # {
        #     "messages": [dict],  # 转译过程中的对话消息
        #     "translation": TranslateResponse,  # 转译结果
        #     "temperature": float,  # 转译时的温度
        #     "check_result": dict,  # 转译后的 Rust 代码编译结果
        #     "has_tried": bool,  # 是否已经尝试过
        # }
        # """
        # self.translation_candidates: list[dict] = []
        # # 当前采纳的转译结果
        # self.translation_completion: Optional[dict] = None
        # # 记忆模块
        # self.long_memory = TranspileMemory(llm_config, state_manager)
        # self.translation_experience = []

        # TODO：暂时不使用工具，对于 DeepSeek 模型，不能保证该模型是否能够在该使用工具的场景下正确使用工具。
        # self.register_tool(modify_file_tool)
        # self.register_tool(read_file_tool)
        # self.register_tool(file_insert_content_tool)

    @property
    def module_translation(self) -> ModuleTranslation:
        module_translation = self.translator.state_manager.get_module_translation_by_id(self.module_translation_id)
        if module_translation is None:
            raise ValueError(f"module translation not found: {self.module_translation_id}")
        return module_translation

    @property
    def translation_task(self) -> TranslationTask:
        module_translation = self.module_translation
        translation_task = module_translation.get_translation_task_by_id(self.translation_task_id)
        if translation_task is None:
            raise ValueError(f"translation task not found: {self.translation_task_id}")
        return translation_task

    def run(self, agent_response: AgentResponse) -> AgentResponse:
        if agent_response.status == AgentResponseStatus.DONE:
            if agent_response.type == CodeMonkeyResponseType.PREPARE_TRANSLATION_TASK:
                # 准备转译任务完成后，开始转译
                return asyncio.run(self.translate(agent_response))
            elif agent_response.type == CodeMonkeyResponseType.TRANSLATION_COMPLETION:
                # 转译完成后，评估转译结果
                return asyncio.run(self.evaluate_translation(agent_response))
            elif agent_response.type == CodeMonkeyResponseType.COMPILE_CHECK_DONE:
                # 编译检查完成后，审查转译结果
                return asyncio.run(self.review_translation(agent_response))
            elif agent_response.type == CodeMonkeyResponseType.COMPILE_CHECK_FAILED:
                # 编译检查失败，修复编译错误
                return asyncio.run(self.fix_compile_errors(agent_response))
            elif agent_response.type in (
                    CodeMonkeyResponseType.FIX_FAILED,
                    CodeMonkeyResponseType.TRANSLATION_COMPLETION_FAILED
            ):
                # 错误修复失败
                return asyncio.run(self.make_as_unimplemented(agent_response))
            else:
                self.logger.error(f"unknown response type: {agent_response.type}")
                return AgentResponse.error(self, CodeMonkeyResponseType.TRANSLATION_TASK_DONE, error={
                    "message": f"unknown response type: {agent_response.type}"
                })
                # return AgentResponse.done(self, CodeMonkeyResponseType.TRANSLATION_TASK_DONE)
            # elif pre_response.type == CodeMonkeyResponseType.TRANSLATION_EXPERIENCE_QUERY:
            #     # 经验查询完成后，开始进行转译
            #     return asyncio.run(self.translate(pre_response))
            # elif pre_response.type == CodeMonkeyResponseType.TRANSLATION_COMPLETION:
            #     # 转译完成后，评估转译结果
            #     return self.evaluate_translation(pre_response)
            # # elif pre_response.type in [CodeMonkeyResponseType.WRITE_CODE, CodeMonkeyResponseType.FIX_CODE]:
            # #     # 写入代码完成，检查代码
            # #     return self.check_code(pre_response)
            # elif pre_response.type == CodeMonkeyResponseType.CHECK_CODE_FAILED:
            #     # 代码检查未通过，根据错误进行反思
            #     return AgentResponse.done(self, CodeMonkeyResponseType.TRANSLATION_TASK_DONE)
            #     return self.fixing(pre_response)
            # elif pre_response.type == CodeMonkeyResponseType.FIX_FAILED:
            #     # 错误修复超出限制
            #     return self.mark_as_todo(pre_response)
        elif agent_response.type == AgentResponseStatus.ERROR:
            # TODO: 错误处理
            self.logger.error(f"error occurred in run: {agent_response}")
            return agent_response

    def start(self, module_translation: ModuleTranslation, translation_task: TranslationTask) -> AgentResponse:
        self.logger.info(f"start Code Monkey [{self.name}] for translation task [{translation_task.id}]")
        agent_response = self.prepare_translation_task(module_translation, translation_task)
        last_response = agent_response
        while True:
            agent_response = self.run(agent_response)
            if agent_response is None:
                return AgentResponse.error(self, CodeMonkeyResponseType.TRANSLATION_TASK_DONE, error={
                    "message": f"no response from the agent, last response: {last_response.type}"
                })
            if agent_response.type == CodeMonkeyResponseType.TRANSLATION_TASK_DONE:
                # 所有转译任务完成
                return agent_response
            last_response = agent_response

    def prepare_translation_task(self, module_translation: ModuleTranslation,
                                 translation_task: TranslationTask) -> AgentResponse:
        self.logger.info(f"preparing translation task: [{translation_task.id}] in module [{module_translation.name}]")
        self.module_translation_id = module_translation.id
        self.translation_task_id = translation_task.id
        return AgentResponse.done(self, CodeMonkeyResponseType.PREPARE_TRANSLATION_TASK)

    async def translate(self, agent_response: AgentResponse) -> AgentResponse:
        """转译代码"""
        # return await self.translate_directly(agent_response)
        return await self.translate_with_reasoner(agent_response)

    async def translate_directly(self, agent_response: AgentResponse) -> AgentResponse:
        """直接转译代码"""
        # 1. 更新转译任务状态为运行中
        await self.translator.state_manager.update_translation_task_status(
            self.module_translation_id, self.translation_task_id, TranslationTaskStatus.TRANSLATING
        )
        # 2. 构建转译任务的上下文提示词
        module_name = self.module_translation.name
        module_description = self.module_translation.description
        module_files = self.translator.state_manager.state.target_project.list_files(
            show_content=True,
            show_line_numbers=True,
            ignore_func=lambda filepath: filepath not in self.module_translation.related_rust_files,
            relpath=self.module_translation.path
        )
        current_translation_task = self.translation_task
        related_translation_tasks = [
            self.module_translation.get_translation_task_by_id(task_id)
            for task_id in self.translation_task.prerequisites
        ]
        related_node_types = list(set([
            node.type
            for node in self.translation_task.source.nodes
        ]))
        translate_directly_prompt = PromptLoader.get_prompt(
            f"{self.ROLE}/translate_directly.prompt",
            module_name=module_name,
            module_description=module_description,
            project_files=module_files,
            show_file_content=True,
            current_translation_task=current_translation_task,
            related_translation_tasks=related_translation_tasks,
            node_types=related_node_types,
            json_schema=TranslateResponse.model_json_schema()
        )
        self.translation_context = PromptLoader.get_prompt(
            f"{self.ROLE}/translation_context.prompt",
            module_name=module_name,
            module_description=module_description,
            project_files=module_files,
            show_file_content=True,
            current_translation_task=current_translation_task,
            related_translation_tasks=related_translation_tasks
        )
        self.logger.debug(
            f"[{self.module_translation.name} | [{self.translation_task.source.name}({self.translation_task.id})] translate_directly prompt: \n{translate_directly_prompt}")
        translate_messages = [
            {"role": "user", "content": translate_directly_prompt}
        ]
        # 3. 直接进行转译
        translate_temperature = 0.01
        max_try_count = 10
        while max_try_count > 0:
            response = self.call_llm(messages=translate_messages, temperature=translate_temperature)
            translate_response_content = response.choices[0].message.content
            translate_messages.append({"role": "assistant", "content": translate_response_content})
            json_formats = parse_json_format(translate_response_content, TranslateResponse)
            translate_response_obj = json_formats[-1] if isinstance(json_formats, list) else json_formats
            if translate_response_obj is None:
                translate_messages.append(
                    {"role": "user", "content": "解析转译结果失败，请严格遵守转译结果的 JSON 格式"})
                max_try_count -= 1
                continue
            else:
                self.translate_messages = [
                    {"role": "user", "content": translate_directly_prompt},
                    {"role": "assistant", "content": translate_response_content}
                ]
                self.logger.debug(
                    f"[{self.module_translation.name} | [{self.translation_task.source.name}({self.translation_task.id})] translate_with_reasoner content: \n{translate_response_content}")
                self.translation_candidate_list.append({
                    "messages": translate_messages,
                    "translation": translate_response_obj,
                    "temperature": translate_temperature,
                    "compile_result": None,
                    "has_tried": False
                })
                return AgentResponse.done(self, CodeMonkeyResponseType.TRANSLATION_COMPLETION)
        return AgentResponse.error(self, CodeMonkeyResponseType.TRANSLATION_COMPLETION_FAILED, error={
            "message": "failed to parse JSON as TranslateResponse."
        })

    async def translate_with_reasoner(self, agent_response: AgentResponse) -> AgentResponse:
        """转译代码"""
        # 1. 更新转译任务状态为运行中
        await self.translator.state_manager.update_translation_task_status(
            self.module_translation_id, self.translation_task_id, TranslationTaskStatus.TRANSLATING
        )
        # 2. 构建转译任务的上下文提示词
        module_name = self.module_translation.name
        module_description = self.module_translation.description
        module_files = self.translator.state_manager.state.target_project.list_files(
            show_content=True,
            show_line_numbers=True,
            ignore_func=lambda filepath: filepath not in self.module_translation.related_rust_files,
            relpath=self.module_translation.path
        )
        current_translation_task = self.translation_task
        related_translation_tasks = [
            self.module_translation.get_translation_task_by_id(task_id)
            for task_id in self.translation_task.prerequisites
        ]
        related_node_types = list(set([
            node.type
            for node in self.translation_task.source.nodes
        ]))
        translate_with_reasoner_prompt = PromptLoader.get_prompt(
            f"{self.ROLE}/translate_with_reasoner.prompt",
            module_name=module_name,
            module_description=module_description,
            project_files=module_files,
            show_file_content=True,
            current_translation_task=current_translation_task,
            related_translation_tasks=related_translation_tasks,
            node_types=related_node_types
        )
        self.translation_context = PromptLoader.get_prompt(
            f"{self.ROLE}/translation_context.prompt",
            module_name=module_name,
            module_description=module_description,
            project_files=module_files,
            show_file_content=True,
            current_translation_task=current_translation_task,
            related_translation_tasks=related_translation_tasks
        )
        self.logger.debug(
            f"[{self.module_translation.name} | [{self.translation_task.source.name}({self.translation_task.id})] translate_with_reasoner prompt: \n{translate_with_reasoner_prompt}")
        translate_messages = [
            {"role": "user", "content": translate_with_reasoner_prompt}
        ]
        # 推理模型默认温度为0.6
        translate_temperature = self.reasoner.llm_config.get("temperature", 0.6)
        # 3. 调用推理模型进行转译
        max_try_count = 10
        while max_try_count > 0:
            response = self.reasoner.call_llm(messages=translate_messages, temperature=translate_temperature,
                                              json_format=TranslateResponse)
            translate_response_content = response.choices[0].message.content
            translate_messages.append({"role": "assistant", "content": translate_response_content})
            translate_response_obj = response.format_object
            if translate_response_obj is None:
                self.logger.debug(
                    f"json parse exception: [{self.module_translation.name} | [{self.translation_task.source.name}({self.translation_task.id})] translate_with_reasoner content: \n{translate_response_content}"
                )
                translate_messages.append(
                    {"role": "user", "content": "解析转译结果失败，请严格遵守转译结果的 JSON 格式。"})
                max_try_count -= 1
                continue
            else:
                self.translate_messages = [
                    {"role": "user", "content": translate_with_reasoner_prompt},
                    {"role": "assistant", "content": translate_response_content}
                ]
                self.logger.debug(
                    f"[{self.module_translation.name} | [{self.translation_task.source.name}({self.translation_task.id})] translate_with_reasoner content: \n{translate_response_content}")
                self.translation_candidate_list.append({
                    "messages": translate_messages,
                    "translation": translate_response_obj,
                    "temperature": translate_temperature,
                    "compile_result": None,
                    "has_tried": False
                })
                return AgentResponse.done(self, CodeMonkeyResponseType.TRANSLATION_COMPLETION)
        return AgentResponse.error(self, CodeMonkeyResponseType.TRANSLATION_COMPLETION_FAILED, error={
            "message": "failed to parse JSON as TranslateResponse."
        })

    async def evaluate_translation(self, agent_response: AgentResponse) -> AgentResponse:
        """评估转译结果"""
        untried_candidates = [
            candidate
            for candidate in self.translation_candidate_list
            if not candidate.get("has_tried", False)
        ]
        best_candidate = None
        if len(untried_candidates) == 0:
            # 所有候选结果均已尝试，但未成功
            return AgentResponse.done(self, CodeMonkeyResponseType.TRANSLATION_COMPLETION_FAILED)
        elif len(untried_candidates) == 1:
            # 只有一个候选结果，直接采纳
            best_candidate = untried_candidates[0]
        else:
            # TODO：暂时先取第一个
            best_candidate = untried_candidates[0]
            # raise NotImplementedError("multiple translation candidates evaluation is not implemented.")
        self.translation_candidate = best_candidate
        self.translation_candidate["has_tried"] = True
        translation_result = best_candidate["translation"]

        if translation_result.type == "comment":
            # 无 Rust 代码，直接返回
            return AgentResponse.done(self, CodeMonkeyResponseType.COMPILE_CHECK_DONE)
        else:
            # 有 Rust 代码，进行编译检查
            if translation_result.filepath not in self.module_translation.related_rust_files:
                # 该 Rust 文件不在相关 Rust 文件列表中，添加到相关 Rust 文件列表中
                await self.translator.state_manager.add_module_translation_related_rust_files(
                    self.module_translation_id, [translation_result.filepath]
                )
            rust_filepath = os.path.join(self.module_translation.path, translation_result.filepath)
            with self.translator.file_lock_manager.file_lock(rust_filepath):
                if not os.path.exists(rust_filepath):
                    # 文件不存在
                    os.makedirs(os.path.dirname(rust_filepath), exist_ok=True)
                    with open(rust_filepath, "w", encoding="utf-8") as f:
                        f.write(translation_result.text)
                else:
                    # 文件存在
                    with open(rust_filepath, "r", encoding="utf-8") as f:
                        temp_rust_file_content = f.read()
                    with open(rust_filepath, "a", encoding="utf-8") as f:
                        if temp_rust_file_content.strip() != "":
                            f.write("\n\n" + translation_result.text)
                            self.translation_candidate[
                                "code"] = temp_rust_file_content + "\n\n" + translation_result.text
                        else:
                            f.write(translation_result.text)
                            self.translation_candidate["code"] = translation_result.text
                compile_check_result = cargo_check(self.module_translation.path, [rust_filepath],
                                                   ignore_codes=["E0601"])
                self.translation_candidate["compile_result"] = compile_check_result
                if compile_check_result["success"]:
                    # 编译通过
                    return AgentResponse.done(self, CodeMonkeyResponseType.COMPILE_CHECK_DONE)
                else:
                    # 编译失败
                    return AgentResponse.done(self, CodeMonkeyResponseType.COMPILE_CHECK_FAILED, data={
                        "compile_errors": compile_check_result["errors"]
                    })

    # async def compile_check(self, agent_response: AgentResponse) -> AgentResponse:
    #     """编译检查"""
    #     # 忽略 E0601 错误：main 函数未找到
    #     cargo_check_result = cargo_check(
    #         self.module_translation.path,
    #         ignore_codes=["E0601"]
    #     )
    #     if cargo_check_result["success"]:
    #         # 编译通过
    #         self.logger.info(f"compile check done")
    #         return AgentResponse.done(self, CodeMonkeyResponseType.COMPILE_CHECK_DONE)
    #     else:
    #         # 编译失败
    #         self.compile_errors = cargo_check_result["errors"]
    #         self.logger.info(f"compile check failed: \n{self.compile_errors}")
    #         return AgentResponse.done(self, CodeMonkeyResponseType.COMPILE_CHECK_FAILED, data={
    #             "compile_errors": self.compile_errors
    #         })

    async def review_translation(self, agent_response: AgentResponse) -> AgentResponse:
        """审查转译，提取经验以及构建转译结果"""
        # 1. 构建转译结果
        translate_result = None
        if self.translation_candidate["compile_result"] is None or self.translation_candidate["compile_result"][
            "success"]:
            # 转译后直接编译通过
            translate_result = self.translation_candidate["translation"]
        else:
            # 经过修复后才通过
            project_files = self.translator.state_manager.state.target_project.list_files(
                show_content=True,
                relpath=self.module_translation.path,
                ignore_func=lambda filepath: filepath not in self.module_translation.related_rust_files
            )
            translation_target_prompt = PromptLoader.get_prompt(
                f"{self.ROLE}/review_translation_target.prompt",
                translation_context=self.translation_context,
                project_files=project_files
            )
            translation_task_target_messages = [
                {"role": "user", "content": translation_target_prompt}
            ]
            max_try_count = 3
            while max_try_count > 0:
                translate_result_response = self.call_llm(
                    messages=translation_task_target_messages,
                    json_format=TranslateResponse,
                )
                translate_result = translate_result_response.format_object
                if translate_result is None:
                    translation_task_target_messages.append({
                        "role": "user",
                        "content": "解析转译结果失败，请严格遵守转译结果的 JSON 格式。"
                    })
                    max_try_count -= 1
                    continue
                else:
                    break
        if translate_result is None:
            ...
        else:
            translation_task_target = TranslationTaskTarget(
                type=translate_result.type,
                name=self.translation_task.source.name,
                text=translate_result.text,
                description=translate_result.explanation,
                filepath=translate_result.filepath
            )
            await self.translator.state_manager.set_translation_task_target(
                self.module_translation_id, self.translation_task_id, translation_task_target
            )
        # 2. TODO：提取转译经验

        await self.translator.state_manager.update_translation_task_status(
            self.module_translation_id, self.translation_task_id, TranslationTaskStatus.DONE
        )
        return AgentResponse.done(self, CodeMonkeyResponseType.TRANSLATION_TASK_DONE)

    async def fix_with_reasoner(self, agent_response: AgentResponse) -> AgentResponse:
        """修复编译错误"""
        # 修复错误时需要保证当前文件没有任何修改，对文件加锁
        for related_file in self.module_translation.related_rust_files:
            filepath = os.path.join(self.module_translation.path, related_file)
            if self.translator.file_lock_manager.is_lock(filepath):
                self.logger.info("file is locked: " + filepath)
                continue
            self.translator.file_lock_manager.acquire_file_lock(filepath)
            self.logger.info("acquire file lock: " + filepath)
        await self.translator.state_manager.update_translation_task_status(
            self.module_translation_id, self.translation_task_id, TranslationTaskStatus.FIXING
        )
        compile_errors = agent_response.data.get("compile_errors", [])
        # 过多错误可能是由于同一位置的错误重复出现，故先处理一部分错误
        if len(compile_errors) >= 20:
            compile_errors = compile_errors[:20]
        # 错误信息
        errors = [
            error["rendered"]
            for error in compile_errors
        ]
        error_document = {}
        for error in compile_errors:
            if error["code"]:
                error_code = error["code"]["code"]
                if error_code not in error_document:
                    error_document[error_code] = error["code"]["explanation"]
        explanations = [
            f"## {error_code}\n{explanation}\n"
            for error_code, explanation in error_document.items()
        ]
        fix_errors_messages = []
        # 最大修复迭代次数
        max_try_count = 10
        while max_try_count > 0:
            project_files = self.translator.state_manager.state.target_project.list_files(
                show_content=True,
                show_line_numbers=True,
                relpath=self.module_translation.path,
                ignore_func=lambda filepath: filepath not in self.module_translation.related_rust_files
            )
            fix_errors_prompt = PromptLoader.get_prompt(
                f"{self.ROLE}/fix_with_reasoner.prompt",
                module_name=self.module_translation.name,
                errors=errors,
                explanations=explanations,
                project_files=project_files,
                show_file_content=True
            )
            fix_errors_messages.append(
                {"role": "user", "content": fix_errors_prompt}
            )
            self.logger.debug(
                f"[{self.module_translation.name} | [{self.translation_task.source.name}({self.translation_task.id})] fix_with_reasoner prompt: \n{fix_errors_prompt}")
            if len(fix_errors_messages) > 3:
                # 只保留最近的三条消息（转译上下文、提问、回答、提问）
                fix_errors_messages = [{
                    "role": "user",
                    "content": self.translation_context
                }] + fix_errors_messages[-3:]
            reasoner_response = self.reasoner.call_llm(messages=fix_errors_messages, temperature=0.6)
            reasoner_response_content = reasoner_response.choices[0].message.content
            reasoner_response_reasoning_content = reasoner_response.choices[0].message.reasoning_content
            fix_errors_messages.append(
                {"role": "assistant", "content": reasoner_response_content}
            )
            self.logger.debug(
                f"[{self.module_translation.name} | [{self.translation_task.source.name}({self.translation_task.id})] fix_with_reasoner content: \n{reasoner_response_content}")
            code_block_change_info = extract_code_block_change_info(reasoner_response_content)
            for file, changes in code_block_change_info.items():
                filepath = os.path.join(self.module_translation.path, file)
                if os.path.exists(filepath):
                    with open(filepath, "r", encoding="utf-8") as f:
                        old_content = f.read()
                else:
                    old_content = ""
                try:
                    new_content = apply_changes(old_content, changes)
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(new_content)
                except Exception as e:
                    self.logger.error(f"failed to apply changes to file: {filepath}, error: {e}")
                    continue
            # 修复完成后，重新进行编译检查
            compile_check_result = cargo_check(self.module_translation.path, ignore_codes=["E0601"])
            if compile_check_result["success"]:
                # 编译通过, 解锁文件
                for related_file in self.module_translation.related_rust_files:
                    filepath = os.path.join(self.module_translation.path, related_file)
                    if self.translator.file_lock_manager.is_lock(filepath):
                        self.translator.file_lock_manager.release_file_lock(filepath)
                        self.logger.info(f"release file lock: {filepath}")
                return AgentResponse.done(self, CodeMonkeyResponseType.COMPILE_CHECK_DONE)
            else:
                # 编译失败
                errors = [
                    error["rendered"]
                    for error in compile_check_result["errors"]
                ]
                self.logger.info(f"compile check failed after fixing errors, try again: \n" + "\n".join(errors))
                error_document = {}
                for error in compile_errors:
                    if error["code"]:
                        error_code = error["code"]["code"]
                        if error_code not in error_document:
                            error_document[error_code] = error["code"]["explanation"]
                explanations = [
                    f"## {error_code}\n{explanation}\n"
                    for error_code, explanation in error_document.items()
                ]
                max_try_count -= 1
                continue
        return AgentResponse.done(self, CodeMonkeyResponseType.FIX_FAILED)

    async def fix_compile_errors(self, agent_response: AgentResponse) -> AgentResponse:
        """修复编译错误"""
        await self.translator.state_manager.update_translation_task_status(
            self.module_translation_id, self.translation_task_id, TranslationTaskStatus.FIXING
        )
        return await self.fix_with_reasoner(agent_response)

    async def make_as_unimplemented(self, agent_response: AgentResponse) -> AgentResponse:
        """标记为未实现"""
        self.logger.info(f"mark {self.translation_task.source.name}({self.translation_task.id}) as unimplemented")
        project_files = self.translator.state_manager.state.target_project.list_files(
            show_content=True,
            show_line_numbers=True,
            relpath=self.module_translation.path,
            ignore_func=lambda filepath: filepath not in self.module_translation.related_rust_files
        )
        make_as_unimplemented_prompt = PromptLoader.get_prompt(
            f"{self.ROLE}/make_as_unimplemented.prompt",
            translation_context=self.translation_context,
            module_name=self.module_translation.name,
            project_files=project_files
        )
        make_as_unimplemented_messages = [
            {"role": "user", "content": make_as_unimplemented_prompt}
        ]
        reasoner_response = self.reasoner.call_llm(messages=make_as_unimplemented_messages, temperature=0.6)
        reasoner_response_content = reasoner_response.choices[0].message.content
        reasoner_response_reasoning_content = reasoner_response.choices[0].message.reasoning_content
        changed_code_blocks = extract_code_block_change_info(reasoner_response_content)
        for file, changes in changed_code_blocks.items():
            filepath = os.path.join(self.module_translation.path, file)
            with open(filepath, "r", encoding="utf-8") as f:
                old_content = f.read()
            try:
                new_content = apply_changes(old_content, changes)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(new_content)
            except Exception as e:
                self.logger.error(f"failed to apply changes to file: {filepath}, error: {e}")
                continue

        compile_result = cargo_check(self.module_translation.path, ignore_codes=["E0601"])
        if compile_result["success"]:
            for related_file in self.module_translation.related_rust_files:
                filepath = os.path.join(self.module_translation.path, related_file)
                if self.translator.file_lock_manager.is_lock(filepath):
                    self.translator.file_lock_manager.release_file_lock(filepath)
                    self.logger.info(f"release file lock: {filepath}")
            return AgentResponse.done(self, CodeMonkeyResponseType.COMPILE_CHECK_DONE)
        else:
            return AgentResponse.done(self, CodeMonkeyResponseType.COMPILE_CHECK_FAILED, data={
                "compile_errors": compile_result["errors"]
            })