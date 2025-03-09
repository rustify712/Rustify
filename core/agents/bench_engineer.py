import os
import re
from typing import Optional

import toml

from core.agents.base import AgentResponseStatus, BaseAgent, AgentResponse, AgentRequest, AgentResponseType
from core.utils.prompt_loader import PromptLoader
from core.schema.response import BenchEngineerResponseType
from core.state.state_manager import ModuleTranslation, StateManager, ModuleTranslationStatus
from core.tools.file_tools import read_file_tool
from core.utils.rust_utils import cargo_bench


class BenchEngineer(BaseAgent):
    """
    生成并执行基准测试代码
    """

    ROLE = "bench_engineer"
    DESCRIPTION = "Write code to test the performance of the model."

    def __init__(
            self,
            llm_config: dict,
            state_manager: "StateManager",
            **kwargs
    ):
        super().__init__(llm_config, **kwargs)
        self.state_manager = state_manager
        self.module_translation: Optional[ModuleTranslation] = None
        # 相关 Rust 文件路径（相对路径）
        self.rust_file = None
        # benchmark 文件路径（相对路径）
        self.bench_file = None
        # benchmark 名
        self.bench_name = None
        # Rust lib 模块名
        self.crate_name = None
        # 修复次数
        self.fixing_count = 0
        self.modified_files = []

    def run(self, pre_response: AgentResponse) -> AgentResponse:
        if pre_response.status == AgentResponseStatus.DONE:
            if pre_response.type == BenchEngineerResponseType.BENCH_PREPARE_DONE:
                return self.generate_benchmark(pre_response)
            elif pre_response.type in [BenchEngineerResponseType.BENCH_COMPLETION, BenchEngineerResponseType.BENCH_FIX_DONE]:
                return self.bench_code(pre_response)
            elif pre_response.type == BenchEngineerResponseType.BENCH_FAILED:
                return self.fixing(pre_response)
        elif pre_response.type == AgentResponseStatus.ERROR:
            ...

    def start(self, module_name: str) -> AgentResponse:
        self.logger.info("Bench Engineer started.")
        agent_response = self.prepare_benchmark(module_name)
        while True:
            agent_response = self.run(agent_response)
            if agent_response is None:
                return AgentResponse.error(self, BenchEngineerResponseType.BENCH_DONE, error={
                    "message": "No response from the agent."
                })
            if agent_response.type == BenchEngineerResponseType.BENCH_DONE:
                return agent_response

    def prepare_benchmark(self, module_name: str) -> AgentResponse:
        """准备基准测试

        1. 初始化基准测试文件
        2. 配置基准测试依赖
        """
        self.logger.info(f"Prepare benchmark for module [{module_name}]...")
        # TODO: 目前只支持一个模块对应一个基准测试文件
        self.module_translation = self.state_manager.translator.module_translations[module_name]
        if not self.module_translation.related_rust_files:
            self.logger.info("No related rust files found.")
            return AgentResponse.done(self, BenchEngineerResponseType.BENCH_DONE)
        self.rust_file = self.module_translation.related_rust_files[0]
        self.bench_name = self.rust_file.split("/")[-1].replace(".rs", "") + "_bench"
        self.bench_file = f"benches/{self.bench_name}.rs"
        os.makedirs(os.path.join(self.state_manager.target_project.path, "benches"), exist_ok=True)
        # 向 Cargo.toml 中写入基准测试依赖
        cargo_toml_filepath = os.path.join(self.state_manager.target_project.path, "Cargo.toml")
        with self.state_manager.file_lock_manager.file_lock(cargo_toml_filepath):
            with open(cargo_toml_filepath, "r", encoding="utf-8") as f:
                cargo_toml = toml.load(f)
            package = cargo_toml.get('package', {})
            self.crate_name = package.get('name')
            dependencies = cargo_toml.get('dependencies', {})
            if "criterion" not in dependencies:
                dependencies["criterion"] = "0.5.1"
            # 向 Cargo.toml 中写入基准测试配置
            benches = cargo_toml.get("bench", [])
            if self.bench_name not in [bench.get("name") for bench in benches]:
                benches.append({
                    "name": self.bench_name,
                    "harness": False
                })
            cargo_toml["bench"] = benches
            with open(cargo_toml_filepath, "w", encoding="utf-8") as f:
                toml.dump(cargo_toml, f)
        return AgentResponse.done(self, BenchEngineerResponseType.BENCH_PREPARE_DONE)

    def generate_benchmark(self, pre_response: AgentResponse) -> AgentResponse:
        """生成基准测试代码"""
        rust_filepath = os.path.join(self.state_manager.target_project.path, self.rust_file)
        module_code = read_file_tool(rust_filepath, show_line_number=False)
        generate_benchmark_prompt = PromptLoader.get_prompt(
            f"{self.ROLE}/generate_benchmark.prompt",
            crate_name=self.crate_name,
            module_name=self.rust_file,
            module_code=module_code,
            project_structure=self.state_manager.target_project.pretty_structure()
        )
        self.logger.debug(f"generate benchmark prompt: \n{generate_benchmark_prompt}")
        generate_benchmark_messages = [{
            "role": "user",
            "content": generate_benchmark_prompt
        }]
        try_count = 0
        while try_count < 3:
            generate_benchmark_response = self.call_llm(generate_benchmark_messages, temperature=0.01)
            generate_benchmark_message_content = generate_benchmark_response.choices[0].message.content
            generate_benchmark_messages.append({"role": "assistant", "content": generate_benchmark_message_content})
            rust_code_blocks = re.findall(r"```rust\s*(.*?)\s*```", generate_benchmark_message_content, re.DOTALL)
            if len(rust_code_blocks) == 0:
                self.logger.error("No Rust code block found in the response.")
                generate_benchmark_messages.append(
                    {"role": "user", "content": "No Rust code block found in the response."})
            else:
                # TODO: 更加复杂的校验规则或利用工具，这里仅仅简单的获取最后一个 Rust 代码块
                bench_code = rust_code_blocks[-1]
                bench_filepath = os.path.join(self.state_manager.target_project.path, self.bench_file)
                if bench_code.strip() == "":
                    self.logger.info("No need to generate benchmark code.")
                    with self.state_manager.file_lock_manager.file_lock(bench_filepath):
                        with open(bench_filepath, "w", encoding="utf-8") as f:
                            f.write("")
                        self.state_manager.target_vcs.add([bench_filepath])
                        self.state_manager.target_vcs.commit(f"Add benchmark code for {self.rust_file}.")
                    return AgentResponse.done(self, BenchEngineerResponseType.BENCH_DONE)
                self.logger.debug(f"Benchmark code generated: \n{bench_code}")
                with self.state_manager.file_lock_manager.file_lock(bench_filepath):
                    with open(bench_filepath, "w", encoding="utf-8") as f:
                        f.write(bench_code)
                    self.state_manager.target_vcs.add([bench_filepath])
                return AgentResponse.done(self, BenchEngineerResponseType.BENCH_COMPLETION)
            try_count += 1
        return AgentResponse.error(self, BenchEngineerResponseType.BENCH_DONE, error={
            "message": "Failed to generate benchmark code."
        })

    def bench_code(self, pre_response: AgentResponse) -> AgentResponse:
        """执行基准测试"""
        self.logger.info(f"Start benchmarking [{self.bench_name}]...")
        bench_output = cargo_bench(self.state_manager.target_project.path, self.bench_name)
        if bench_output["success"]:
            self.logger.info(f"[{self.bench_name}] benchmark done.")
            self.state_manager.target_vcs.commit(f"Add benchmark code for {self.rust_file}.")
            # 记录 benchmark 结果, Rust 模块名 -> benchmark 名
            self.state_manager.add_module_bench_files(self.module_translation.module_name, [self.bench_file])
            self.state_manager.mark_module_translation_as(self.module_translation.module_name, ModuleTranslationStatus.BENCHMARK)
            return AgentResponse.done(self, BenchEngineerResponseType.BENCH_DONE)
        else:
            self.logger.debug("benchmark occurred errors: \n" + "\n".join([error["rendered"] for error in bench_output["errors"]]))
            return AgentResponse.done(self, BenchEngineerResponseType.BENCH_FAILED, data={
                "errors": bench_output["errors"]
            })

    def fixing(self, pre_response: AgentResponse) -> AgentResponse:
        if self.fixing_count >= 10:
            return AgentResponse.error(self, BenchEngineerResponseType.BENCH_DONE, error={
                "message": "Failed to fix the code with feedback: the fixing count has reached the limit."
            })
        errors = pre_response.data.get("errors")

        project_files = self.state_manager.target_project.list_files(
            show_content=True,
            ignore_func=lambda filepath: os.path.isfile(os.path.join(self.state_manager.target_project.path, filepath)) and filepath not in (
                "src/lib.rs", self.rust_file, self.bench_file
            )
        )
        fixing_prompt = PromptLoader.get_prompt(
            f"{self.ROLE}/fixing.prompt",
            module_filepath=self.rust_file,
            bench_filepath=self.bench_file,
            errors=errors,
            project_files=project_files,
            is_file_content=True
        )
        self.logger.debug(f"fix prompt: \n{fixing_prompt}")
        fix_with_feedback_messages = [
            {"role": "user", "content": fixing_prompt}
        ]
        try_count = 0
        while try_count < 3:
            response = self.call_llm(fix_with_feedback_messages, temperature=0.2)
            fix_with_feedback_message_content = response.choices[0].message.content
            self.logger.debug(f"fix response content: \n{fix_with_feedback_message_content}")
            fix_with_feedback_messages.append({"role": "assistant", "content": fix_with_feedback_message_content})
            rust_code_blocks = re.findall(r"```rust\s*(.*?)\s*```", fix_with_feedback_message_content, re.DOTALL)
            if len(rust_code_blocks) == 0:
                self.logger.error("No Rust code block found in the response.")
                fix_with_feedback_messages.append(
                    {"role": "user", "content": "No Rust code block found in the response."})
            else:
                bench_filepath = os.path.join(self.state_manager.target_project.path, self.bench_file)
                rust_code = rust_code_blocks[-1]
                # 第一行为文件路径注释
                rust_code_filepath = rust_code.split("\n")[0].strip()
                if "filepath" in rust_code_filepath:
                    rust_code_filepath = os.path.join(self.state_manager.target_project.path, rust_code_filepath.split("filepath:")[1].strip())
                    rust_code_lines = rust_code.split("\n")[1:]
                    with self.state_manager.file_lock_manager.file_lock(rust_code_filepath):
                        with open(rust_code_filepath, "w", encoding="utf-8") as f:
                            f.write("\n".join(rust_code_lines))
                        self.state_manager.target_vcs.add([rust_code_filepath])
                    self.fixing_count += 1
                    return AgentResponse.done(self, BenchEngineerResponseType.BENCH_FIX_DONE)
                else:
                    fix_with_feedback_messages.append(
                        {"role": "user", "content": "No file path found in the code block."})
            try_count += 1
        return AgentResponse.error(self, BenchEngineerResponseType.BENCH_DONE, error={
            "message": "Failed to fix the code with feedback."
        })
