import json
import os
import re
import selectors
from typing import Optional

import toml
import time
from core.agents.base import BaseAgent, AgentResponse, AgentResponseStatus
import subprocess
from core.utils.prompt_loader import PromptLoader
from core.state.state_manager import StateManager
from core.tools.file_tools import read_file_tool
from core.agents.base import AgentRequest, AgentResponse, BaseAgent, AgentResponseType
from core.schema.response import TestEngineerResponseType
from core.utils.rust_utils import cargo_test


class TestEngineer(BaseAgent):
    ROLE = "test_engineer"
    DESCRIPTION = "A powerful and efficient AI assistant that completes tasks accurately and resourcefully, providing translating and generating Rust Test Cases."

    def __init__(
            self,
            llm_config: dict,
            state_manager: "StateManager",
            **kwargs
    ):
        super().__init__(llm_config, **kwargs)
        self.state_manager = state_manager
        self.c_module_file = None
        self.rust_test_file = None
        self.rust_module_file = None
        self.rust_project_path = None
        self.module_name = None
        self.test_name = None
        self.related_rust_files = None
        self.crate_name = None
        self.rust_test_code_path = None
        self.rust_code_path = None
        self.c_test_code_path = None
        self.c_project_path = None
        self.pre_test_case_name = None
        self.modified_files = []
        self.syntax_max_turn = 10
        self.syntax_current_turn = 0
        self.logic_max_turn = 10
        self.logic_current_turn = 0
        # TODO TEMP
        self.fix_logic_with_feedback_messages = []
        self.fix_timeout_messages = []
        self.rust_module_names = None
        self.rust_test_report = None
        self.module_translations = None
        self.change_history = None
        self.history_index = 0
        self.raw_files = None

    def run(self, pre_response: AgentResponse) -> AgentResponse:
        if pre_response.status == AgentResponseStatus.DONE:
            if pre_response.type == TestEngineerResponseType.TEST_PREPARE_DONE:
                # 完成测试准备工作，考试转译测试代码
                return self.translate_test_code(pre_response)
            elif pre_response.type in [TestEngineerResponseType.TEST_CODE_TRANSLATION_COMPLETION,
                                       TestEngineerResponseType.TEST_SYNTAX_FIX_COMPLETION,
                                       TestEngineerResponseType.TEST_LOGIC_FIX_COMPLETION]:
                # 在完成测试代码转译、测试语法修复、测试逻辑修复后均需要运行测试
                return self.run_test(pre_response)
            elif pre_response.type == TestEngineerResponseType.TEST_RUN_FAILURE:
                # 测试运行失败需修复语法错误
                return self.fix_syntax_errors(pre_response)
            elif pre_response.type == TestEngineerResponseType.TEST_RUN_DONE:
                # 测试运行成功，尝试修复不通过的测试用例
                return self.fix_logic_errors(pre_response)
        elif pre_response.type == AgentResponseStatus.ERROR:
            ...

    def start(self, module_name, pre_response: AgentResponse) -> AgentResponse:
        self.logger.info("Test Engineer started.")
        agent_response = self.prepare_test_translation(module_name, pre_response)
        # TODO 删除该语句，当前只是为了跳过生成测试用例的过程
        # agent_response = self.run_test(agent_response)
        while True:
            agent_response = self.run(agent_response)
            if agent_response is None:
                return AgentResponse.error(self, TestEngineerResponseType.TEST_PASSED, error={
                    "message": "No response for agent."
                })
            if agent_response.type == TestEngineerResponseType.TEST_PASSED:
                return agent_response
            if agent_response.type == TestEngineerResponseType.TEST_SYNTAX_FIX_COMPLETION:
                self.syntax_current_turn += 1
            if agent_response.type == TestEngineerResponseType.TEST_LOGIC_FIX_COMPLETION:
                self.syntax_current_turn = 0
                self.logic_current_turn += 1
            if (agent_response.type == TestEngineerResponseType.TEST_RUN_FAILURE
                    and self.syntax_current_turn == self.syntax_max_turn):
                self.logger.info(
                    f"Reached the maximum number of syntax error correction attempts: {self.syntax_max_turn}")
                # 某一次测试运行失败且达到最大语法修错次数，即无法完成语法错误修复
                return AgentResponse.done(self, TestEngineerResponseType.TEST_RUN_FAILURE)
            if (agent_response.type == TestEngineerResponseType.TEST_RUN_DONE
                    and self.logic_current_turn == self.logic_max_turn):
                self.logger.info(
                    f"Reached the maximum number of logical error correction attempts：{self.logic_max_turn}")
                # 某一次测试运行成功且没有通过所有测试用例，即无法完成逻辑错误修复
                return AgentResponse.done(self, TestEngineerResponseType.TEST_FAILED)

    def prepare_test_translation(self, module_name, pre_response: AgentResponse) -> AgentResponse:
        self.syntax_current_turn = 0
        # rust项目根目录
        self.rust_project_path = self.state_manager.target_project.path

        self.module_translations = self.state_manager.translator.module_translations[module_name]
        # c 模块名
        c_module_name = os.path.basename(module_name)
        # 借助LLM选择最合适的文件
        self.c_module_file = self.get_c_test_filepath(module_name)

        self.related_rust_files = list(self.module_translations.related_rust_files)
        # 如果当前模块无对应rust文件，则当前模块无需转译
        if len(self.related_rust_files) == 0:
            return AgentResponse.done(self, TestEngineerResponseType.TEST_PASSED)

        # rust模块名
        self.rust_module_names = [
            os.path.basename(file).replace(".rs", "")
            for file in self.related_rust_files
        ]

        rust_module_name = c_module_name.replace("-", "_")

        self.test_name = f"{rust_module_name}_tests"
        self.rust_test_file = f"tests/{self.test_name}.rs"
        self.rust_test_report = f"reports/tests/{self.test_name}"

        # 初始化tests目录
        os.makedirs(os.path.join(self.rust_project_path, "tests"), exist_ok=True)

        # 初始化tests/report目录
        os.makedirs(os.path.join(self.rust_project_path, self.rust_test_report), exist_ok=True)

        if not self.related_rust_files:
            self.logger.info("No related rust files found.")
            return AgentResponse.error(self, TestEngineerResponseType.TEST_RUN_DONE, error={
                "message": "No related rust files found."
            })

        self.c_project_path = self.state_manager.source_project.path

        # 保存raw_files
        self.raw_files = []
        for filepath in self.related_rust_files:
            self.raw_files.append({
                "filepath": filepath,
                "content": read_file_tool(os.path.join(self.state_manager.target_project.path, filepath),
                                          show_line_number=False)
            })

        # 初始化修改历史
        self.change_history = []

        # 为 Cargo.toml 添加测试依赖
        cargo_toml_filepath = os.path.join(self.state_manager.target_project.path, "Cargo.toml")
        with self.state_manager.file_lock_manager.file_lock(cargo_toml_filepath):
            with open(cargo_toml_filepath, "r", encoding="utf-8") as f:
                cargo_toml = toml.load(f)
            package = cargo_toml.get('package', {})
            self.crate_name = package.get('name')
            dependencies = cargo_toml.get('dependencies', {})
            if "rand" not in dependencies:
                dependencies["rand"] = "0.8"
            with open(cargo_toml_filepath, "w", encoding="utf-8") as f:
                toml.dump(cargo_toml, f)
        return AgentResponse.done(self, TestEngineerResponseType.TEST_PREPARE_DONE)

    def get_c_test_filepath(self, module_name):
        """根据模块代码寻找对应的测试文件"""

        def ignore_func(filepath):
            # TODO 实际上可以去除这一部分，直接在所有文件中选择即可
            if os.path.basename(filepath).startswith("test-"):
                return False
            else:
                return True

        project_files = self.state_manager.source_project.list_files(show_summary=True, ignore_func=ignore_func)
        c_module_filepath = os.path.join(self.state_manager.source_project.path, module_name + ".c")
        c_code = read_file_tool(c_module_filepath)
        find_tests_prompt = PromptLoader.get_prompt(
            f"{self.ROLE}/find_tests.prompt",
            project_files=project_files,
            is_file_content=False,
            show_file_summary=True,
            c_code=c_code,
        )
        self.logger.debug(f"寻找测试文件的prompt：{find_tests_prompt}")
        response = self.call_llm(messages=[{"role": "user", "content": find_tests_prompt}])
        find_tests_content = response.choices[0].message.content
        self.logger.debug(f"寻找测试文件的response：{find_tests_content}")
        json_blocks = re.findall(r"```json\s*(.*?)\s*```", find_tests_content, re.DOTALL)
        if len(json_blocks) == 0:
            self.logger.error("No Rust code block found in the response.")
            return ""
        else:
            test_filepath_dict = json.loads(json_blocks[-1])
            return test_filepath_dict.get("test_file_path")

    def translate_test_code(self, pre_response: AgentResponse) -> AgentResponse:
        c_test_filepath = os.path.join(self.state_manager.source_project.path, self.c_module_file)
        c_test_code = read_file_tool(c_test_filepath, show_line_number=False)

        related_rust_files = self.state_manager.target_project.list_files(
            show_content=True,
            ignore_func=lambda filepath: os.path.isfile(os.path.join(self.state_manager.target_project.path, filepath)) and filepath not in self.related_rust_files + ["src/lib.rs", "src/main.rs", "Cargo.toml"]
        )
        translate_prompt = PromptLoader.get_prompt(
            f"{self.ROLE}/translate.prompt",
            project_files=related_rust_files,
            c_test_code=c_test_code,
            rust_module_names=self.rust_module_names,
            crate_name=self.crate_name,
            is_file_content=True
        )
        translate_test_messages = [{
            "role": "user",
            "content": translate_prompt
        }]
        self.logger.debug(f"转译测试代码的prompt为：{translate_prompt}")
        try_count = 0
        while try_count < 3:
            response = self.call_llm(messages=translate_test_messages)
            translate_test_message_content = response.choices[0].message.content
            translate_test_messages.append({
                "role": "assistant",
                "content": translate_test_message_content
            })
            self.logger.debug(f"转译响应为：{translate_test_message_content}")
            code_blocks = re.findall(r'```rust(.*?)```', translate_test_message_content, re.DOTALL)
            if len(code_blocks) == 0:
                self.logger.error("No Rust Code block found in the response.")
                translate_test_messages.append({"role": "user", "content": "No Rust Code found in the response."})
            else:
                rust_test_code = code_blocks[-1]
                if rust_test_code.strip() == "":
                    self.logger.info("No need to translate the test code.")
                    return AgentResponse.done(self, TestEngineerResponseType.TEST_PASSED)
                self.logger.debug(f"提取得到的rust测试代码为：{rust_test_code}")
                rust_test_code_filepath = os.path.join(self.state_manager.target_project.path, self.rust_test_file)
                # 写入生成的测试代码
                with open(rust_test_code_filepath, "w", encoding="utf-8") as f:
                    f.write(rust_test_code)
                self.state_manager.target_vcs.add([rust_test_code_filepath])
                return AgentResponse.done(self, TestEngineerResponseType.TEST_CODE_TRANSLATION_COMPLETION)
        return AgentResponse.error(self, TestEngineerResponseType.TEST_CODE_TRANSLATION_COMPLETION,
                                   error={"message": "Failed to parse Rust Code."})

    def fix_syntax_errors(self, pre_response: AgentResponse) -> AgentResponse:
        """
        针对测试代码产生的语法错误进行修复
        """
        all_errors = pre_response.data.get("errors")
        errors = [{"index": index + 1, **error} for index, error in enumerate(all_errors) if
                  error.get("level") == "error"]
        self.logger.debug("当前语法错误为：\n")
        for error in errors:
            self.logger.debug(error.get("rendered"))
        rust_test_code_filepath = os.path.join(self.state_manager.target_project.path, self.rust_test_file)

        def ignore_func(filepath):
            # 只保留测试代码和被测代码文件
            return filepath not in self.related_rust_files + [self.rust_test_file, "src/lib.rs", "Cargo.toml"]

        project_files = self.state_manager.target_project.list_files(show_content=True, ignore_func=ignore_func)
        fix_with_feedback_prompt = PromptLoader.get_prompt(
            f"{self.ROLE}/fix_with_feedback.prompt",
            related_rust_files=self.related_rust_files,
            test_filepath=self.rust_test_file,
            errors=errors,
            project_files=project_files,
            is_file_content=True,
            crate_name=self.crate_name
        )
        self.logger.debug(f"fix_with_feedback_prompt为：\n{fix_with_feedback_prompt}")

        fix_with_feedback_messages = [
            {"role": "user", "content": fix_with_feedback_prompt}
        ]
        try_count = 0
        while try_count < 3:
            response = self.call_llm(fix_with_feedback_messages, temperature=0.2)
            fix_with_feedback_message_content = response.choices[0].message.content
            self.logger.debug(f"fix_with_feedback_message_content: {fix_with_feedback_message_content}")
            fix_with_feedback_messages.append({"role": "assistant", "content": fix_with_feedback_message_content})
            rust_code_blocks = re.findall(r"```rust\s*(.*?)\s*```", fix_with_feedback_message_content, re.DOTALL)
            if len(rust_code_blocks) == 0:
                self.logger.error("No Rust code block found in the response.")
                fix_with_feedback_messages.append(
                    {"role": "user", "content": "No Rust code block found in the response."})
            else:
                rust_code = rust_code_blocks[-1]
                # 第一行为文件路径注释
                rust_code_filepath = rust_code.split("\n")[0].strip()
                if "filepath" in rust_code_filepath:
                    rust_code_filepath = os.path.join(self.state_manager.target_project.path,
                                                      rust_code_filepath.split(":")[1].strip())
                    rust_code_lines = rust_code.split("\n")[1:]
                    os.makedirs(os.path.dirname(rust_code_filepath), exist_ok=True)
                    with open(rust_code_filepath, "w", encoding="utf-8") as f:
                        f.write("\n".join(rust_code_lines))
                    self.state_manager.target_vcs.add([rust_code_filepath])
                    if rust_code_filepath not in self.modified_files:
                        self.modified_files.append(rust_code_filepath)
                    self.logger.debug(f"Code fixed {rust_test_code_filepath}: \n{rust_code}")
                    return AgentResponse.done(self, TestEngineerResponseType.TEST_SYNTAX_FIX_COMPLETION)
                else:
                    fix_with_feedback_messages.append(
                        {"role": "user", "content": "No file path found in the code block."})
            try_count += 1
        # 将所有修改的文件恢复
        self.state_manager.target_vcs.revert(self.modified_files)
        return AgentResponse.error(self, TestEngineerResponseType.TEST_SYNTAX_FIX_COMPLETION,
                                   error={
                                       "message": "Failed to fix the syntax error with feedback."
                                   })

    def run_test(self, pre_response: AgentResponse) -> AgentResponse:
        """
        执行测试用例
        """
        try:
            test_output = cargo_test(self.rust_project_path, self.test_name)
            if test_output["success"]:
                # 测试编译通过
                self.logger.info("Compiling test passed.")
                # TODO 在全部测试通过或者达到最大逻辑错误修复次数时才写入
                self.state_manager.add_module_test_files(self.module_translations.module_name, [self.rust_test_file])
                failed_tests = test_output.get("errors")
                if len(failed_tests) == 0:
                    # 测试用例全部通过，无需进行逻辑错误修复
                    self.logger.info("All test cases passed.")
                    self.generate_test_report(test_output["output"])
                    self.add_history()
                    return AgentResponse.done(self, TestEngineerResponseType.TEST_PASSED)
                self.add_history(logic_errors=failed_tests)
                if self.logic_current_turn == self.logic_max_turn:
                    self.generate_test_report(test_output["output"])
                    self.save_best()
                return AgentResponse.done(self, TestEngineerResponseType.TEST_RUN_DONE, data={
                    "test_output": test_output
                })
            else:
                # 测试编译未通过
                # 回到上一次 commit 的版本（撤销写入的代码）
                self.logger.info("Compiling test failed.")
                self.add_history(syntax_errors=test_output["errors"])
                if self.syntax_current_turn == self.syntax_max_turn:
                    # 未能修复编译错误，也写入测试报告
                    rendered_info = []
                    for error in test_output["errors"]:
                        rendered_info.append(error["rendered"])
                    self.generate_test_report("\n".join(rendered_info))
                    # 恢复到最初的版本
                    self.revert_raw_files()
                errors = test_output["errors"]
                return AgentResponse.done(self, TestEngineerResponseType.TEST_RUN_FAILURE, data={
                    "errors": errors
                })
        except Exception as e:
            # TODO: 错误处理
            self.logger.error(f"Error occurred when running code check: {e}")

    def generate_test_report(self, test_console_output):
        """
        生成测试报告
        """
        self.logger.debug(f"测试结果的控制台输出为：{test_console_output}")
        rust_test_code_filepath = os.path.join(self.state_manager.target_project.path, self.rust_test_file)

        def ignore_func(filepath):
            return filepath not in self.related_rust_files and filepath != rust_test_code_filepath

        project_files = self.state_manager.target_project.list_files(show_content=True, ignore_func=ignore_func)

        report_prompt = PromptLoader.get_prompt(
            f"{self.ROLE}/report.prompt",
            project_files=project_files,
            is_file_content=True,
            related_rust_files=self.related_rust_files,
            test_filepath=rust_test_code_filepath,
            console_output=test_console_output
        )
        response = self.call_llm(messages=[{"role": "user", "content": report_prompt}])
        report_content = response.choices[0].message.content
        test_report_filepath = os.path.join(self.state_manager.target_project.path,
                                            f"{self.rust_test_report}/{self.test_name}_report.md")
        with open(test_report_filepath, "w", encoding="utf-8") as f:
            f.write(report_content)
        self.state_manager.target_vcs.add([test_report_filepath])
        self.logger.info("Test report generated.")

    def fix_logic_errors(self, pre_response: AgentResponse) -> AgentResponse:
        """
        基于测试失败信息，修复原rust代码的逻辑错误
        1. 基于prompt思维链的方法
        2. 结合rust-gdb调试信息的方法
        """
        # 获取测试失败的测试函数（以测试函数为修复单位）
        test_output = pre_response.data.get("test_output")
        failed_tests = test_output.get("errors")
        # 获取rust测试代码
        rust_test_code_filepath = os.path.join(self.state_manager.target_project.path, self.rust_test_file)

        # 只保留最近两轮的对话记录
        fix_logic_with_feedback_messages = self.fix_logic_with_feedback_messages[-6:] \
            if len(self.fix_logic_with_feedback_messages) > 6 else self.fix_logic_with_feedback_messages
        fix_timeout_messages = self.fix_timeout_messages[-6:] \
            if len(self.fix_timeout_messages) > 6 else self.fix_timeout_messages

        def ignore_func(filepath):
            return filepath not in self.related_rust_files + [self.rust_test_file, "src/lib.rs", "Cargo.toml"]

        project_files = self.state_manager.target_project.list_files(show_content=True, ignore_func=ignore_func)

        # 每一轮只修复一个测试用例，且修复某个测试用例后才修复下一个
        if test_output.get("timeout"):
            fix_timeout_prompt = PromptLoader.get_prompt(
                f"{self.ROLE}/fix_timeout.prompt",
                project_files=project_files,
                is_file_content=True,
                related_rust_files=self.related_rust_files,
                test_filepath=self.rust_test_file,
                output=test_output.get("output")
            )
            self.logger.debug(f"fix timeout的prompt为：{fix_timeout_prompt}")
            fix_timeout_messages.append({"role": "user", "content": fix_timeout_prompt})
            current_messages = fix_timeout_messages
        else:
            current_test_case = next(
                (failed_test for failed_test in failed_tests
                 if failed_test.get("test_case_name") == self.pre_test_case_name),
                failed_tests[0])
            current_test_case_name = current_test_case.get("test_case_name")
            self.pre_test_case_name = current_test_case_name

            # 判断测试用例是否有问题
            judge_test_prompt = PromptLoader.get_prompt(
                f"{self.ROLE}/judge_test.prompt",
                test_case_name=current_test_case.get("test_case_name"),
                project_files=project_files,
                is_file_content=True,
                related_rust_files=self.related_rust_files,
                test_filepath=rust_test_code_filepath,
                error_info=current_test_case.get("error_info"),
            )
            self.logger.debug(f"judge_test_prompt: {judge_test_prompt}")
            fix_logic_with_feedback_messages.append({"role": "user", "content": judge_test_prompt})
            response = self.call_llm(messages=fix_logic_with_feedback_messages)
            judge_test_content = response.choices[0].message.content
            self.logger.debug(f"judge_test_content: {judge_test_content}")
            fix_logic_with_feedback_messages.append({"role": "assistant", "content": judge_test_content})

            fix_logic_with_feedback_prompt = PromptLoader.get_prompt(
                f"{self.ROLE}/fix_logic_with_feedback.prompt",
                test_case_name=current_test_case.get("test_case_name"),
                project_files=project_files,
                is_file_content=True,
                related_rust_files=self.related_rust_files,
                test_filepath=rust_test_code_filepath,
                error_info=current_test_case.get("error_info"),
                # 若到了指定轮次依然未修复错误，则考虑引入特殊规则重构代码
                rebuild=(self.logic_current_turn == 6)
            )
            self.logger.debug(f"fix logic with feedback的prompt为：{fix_logic_with_feedback_prompt}")
            fix_logic_with_feedback_messages.append({"role": "user", "content": fix_logic_with_feedback_prompt})
            current_messages = fix_logic_with_feedback_messages
        try_count = 0
        while try_count < 3:
            response = self.call_llm(messages=current_messages)
            fix_response_content = response.choices[0].message.content
            self.logger.debug(f"fix logic error的响应为：{fix_response_content}")
            fix_logic_with_feedback_messages.append(
                {"role": "assistant", "content": fix_response_content})
            rust_code_blocks = re.findall(r"```rust\s*(.*?)\s*```", fix_response_content, re.DOTALL)
            if len(rust_code_blocks) == 0:
                self.logger.error("No Rust code block found in the response.")
                fix_logic_with_feedback_messages.append(
                    {"role": "user", "content": "No Rust code block found in the response."})
            else:
                rust_code = rust_code_blocks[-1]
                # 第一行为文件路径注释
                rust_code_filepath = rust_code.split("\n")[0].strip()
                if "filepath" in rust_code_filepath:
                    rust_code_filepath = os.path.join(self.state_manager.target_project.path,
                                                      rust_code_filepath.split(":")[1].strip())
                    rust_code_lines = rust_code.split("\n")[1:]
                    with open(rust_code_filepath, "w", encoding="utf-8") as f:
                        f.write("\n".join(rust_code_lines))
                    self.state_manager.target_vcs.add([rust_code_filepath])
                    if rust_code_filepath not in self.modified_files:
                        self.modified_files.append(rust_code_filepath)
                    self.logger.debug(f"Code fixed {rust_test_code_filepath}: \n{rust_code}")
                    # 根据是否超时更新不同的messages
                    if test_output.get("timeout"):
                        self.fix_timeout_messages = current_messages
                    else:
                        self.fix_logic_with_feedback_messages = fix_logic_with_feedback_messages
                    return AgentResponse.done(self, TestEngineerResponseType.TEST_LOGIC_FIX_COMPLETION)
                else:
                    fix_logic_with_feedback_messages.append(
                        {"role": "user", "content": "No file path found in the code block."})
            try_count += 1
        return AgentResponse.error(self, TestEngineerResponseType.TEST_LOGIC_FIX_COMPLETION,
                                   error={
                                       "message": "Failed to fix the logic error with feedback."
                                   })

    def add_history(self, syntax_errors=None, logic_errors=None):
        """
        维护修改历史
        """
        if syntax_errors is None:
            syntax_errors = []
        if logic_errors is None:
            logic_errors = []

        # 如果编译通过，但是有未通过的测试用例，那么就选择一个最少的恢复
        related_files = []
        for filepath in self.related_rust_files:
            related_files.append({
                "filepath": filepath,
                "content": read_file_tool(os.path.join(self.state_manager.target_project.path, filepath),
                                          show_line_number=False)
            })
        self.change_history.append({
            "syntax_errors": syntax_errors,
            "logic_errors": logic_errors,
            "related_files": related_files,
            "test_content": read_file_tool(os.path.join(self.state_manager.target_project.path, self.rust_test_file),
                                           show_line_number=False)
        })

    def save_best(self):
        """
        如果编译通过，存在不通过的测试用例，则保存最好的结果。
        """
        # 寻找最优历史记录
        best_history_index = 0
        min_errors = len(self.change_history[0].get("logic_errors"))
        for index, history in enumerate(self.change_history):
            errors_num = len(history.get("logic_errors"))
            if errors_num < min_errors:
                best_history_index = index
                min_errors = errors_num
        # 恢复为最优历史记录
        best_history = self.change_history[best_history_index]
        related_files = best_history.get("related_files")
        test_content = best_history.get("test_content")
        for file in related_files:
            filepath = os.path.join(self.state_manager.target_project.path, file.get("filepath"))
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(file.get("content"))
        test_filepath = os.path.join(self.state_manager.target_project.path, self.rust_test_file)
        with open(test_filepath, "w", encoding="utf-8") as f:
            f.write(test_content)

    def revert_raw_files(self):
        """
        如果测试编译未通过，则恢复到最初的版本
        """
        # 恢复最初的被测代码
        for file in self.raw_files:
            filepath = os.path.join(self.state_manager.target_project.path, file.get("filepath"))
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(file.get("content"))
        test_filepath = os.path.join(self.state_manager.target_project.path, self.rust_test_file)
        # 测试文件置为空
        with open(test_filepath, "w", encoding="utf-8") as f:
            f.write("")
