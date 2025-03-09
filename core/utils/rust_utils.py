import json
import os
import re
import time
import subprocess
import selectors
from typing import Any, Dict, Iterable, Optional

try:
    from core.config import Config

    RUSTC_BIN = Config.RUSTC_BIN
    CARGO_BIN = Config.CARGO_BIN
except ImportError:
    RUSTC_BIN = "rustc"
    CARGO_BIN = "cargo"

def deduplicate_by_key(items: Iterable, key=None):
    """
    根据指定的键对列表中的元素进行去重。

    Args:
        items: 需要去重的列表，列表中的元素可以是字典或其他可哈希对象。
        key: 用于去重的键，可以是字典的键名或可调用对象。
    """
    seen = set()
    result = []

    for item in items:
        # 获取用于去重的值
        value = key(item) if key else item

        # 如果该值没有出现过，则添加到结果列表中
        if value not in seen:
            seen.add(value)
            result.append(item)

    return result


def cargo_new_project(project_path: str, project_type: str = "bin") -> str:
    """利用 cargo new 命令创建 Rust 项目
    Args:
        project_path: 项目路径
        project_type: 项目类型，可以是 'bin' 或 'lib'
    """
    try:
        cargo_new_process = subprocess.run(
            [CARGO_BIN, "new", f"--{project_type}", f"{project_path}"],
            capture_output=True,
            text=True
        )
        if cargo_new_process.returncode != 0:
            return cargo_new_process.stderr
        if project_type == "bin":
            # 删除生成的 main.rs 的内容
            with open(f"{project_path}/src/main.rs", "w") as f:
                f.write("fn main() {\n\n}")
        elif project_type == "lib":
            # 删除生成的 lib.rs 中的内容
            with open(f"{project_path}/src/lib.rs", "w") as f:
                f.write("")
        return cargo_new_process.stderr
    except Exception as e:
        raise e


def rustc_check(filepath: str, ignore_codes: Optional[list[str]] = None) -> dict:
    """运行 rustc 命令进行代码检查"""
    ignore_codes = ignore_codes or []
    rustc_check_command = [RUSTC_BIN, filepath, "--error-format=json"]
    try:
        rustc_check_process = subprocess.run(
            rustc_check_command,
            capture_output=True,
            text=True,
        )
        rustc_check_stdout = rustc_check_process.stdout
        output_lines = rustc_check_process.stderr.split("\n")
        compile_errors = []

        for output_line in output_lines:
            if output_line.strip() == "":
                continue
            try:
                rustc_check_output = json.loads(output_line)
                if rustc_check_output["level"] != "error":
                    continue
                if rustc_check_output["code"] and rustc_check_output["code"]["code"] in ignore_codes:
                    continue
                if "aborting due to" in rustc_check_output["message"]:
                    continue
                # TODO：这里目前忽略模块信息，仅针对单模块进行检查
                compile_errors.append({
                    "rendered": rustc_check_output["rendered"],
                    "message_type": rustc_check_output["$message_type"],
                    "children": rustc_check_output["children"],
                    "code": rustc_check_output["code"],
                    "level": rustc_check_output["level"],
                    "message": rustc_check_output["message"],
                    "spans": rustc_check_output["spans"],
                })
            except Exception as e:
                # TODO: 解析错误，详细错误处理
                raise e
        return {
            "success": len(compile_errors) == 0,
            "output": rustc_check_stdout,
            "errors": compile_errors,
        }

    except Exception as e:
        # TODO: 解析错误，详细错误处理
        raise e


def cargo_check(project_dir, filepaths: Optional[list[str]] = None, ignore_codes: Optional[list[str]] = None) -> Dict[str, Any]:
    """运行 cargo check 命令进行代码检查"""
    ignore_codes = ignore_codes or []
    # 要在 os.chdir 之前转为绝对路径
    filepaths = [
        os.path.relpath(filepath, project_dir)
        for filepath in filepaths
    ] if filepaths is not None else []
    cargo_check_command = [CARGO_BIN, "check", "--message-format", "json"]
    try:
        cargo_check_process = subprocess.run(
            cargo_check_command,
            capture_output=True,
            text=True,
            cwd=os.path.abspath(project_dir)
        )
        cargo_check_stderr = cargo_check_process.stderr
        output_lines = cargo_check_process.stdout.split("\n")
        compile_errors = []

        for output_line in output_lines:
            if output_line.strip() == "":
                continue
            try:
                cargo_check_output = json.loads(output_line)
                if cargo_check_output["reason"] == "compiler-message":
                    compiler_message = cargo_check_output
                    if compiler_message["message"]["level"] != "error":
                        continue
                    if compiler_message["message"]["code"] and compiler_message["message"]["code"]["code"] in ignore_codes:
                        continue
                    # Rust 文件相对路径
                    is_record = False
                    for span in compiler_message["message"]["spans"]:
                        if not filepaths:
                            is_record = True
                            break
                        if span["file_name"] in filepaths:
                            is_record = True
                            break
                    if not is_record:
                        continue
                    # TODO：这里目前忽略模块信息，仅针对单模块进行检查
                    compile_errors.append({
                        "rendered": compiler_message["message"]["rendered"],
                        "message_type": compiler_message["message"]["$message_type"],
                        "children": compiler_message["message"]["children"],
                        "code": compiler_message["message"]["code"],
                        "level": compiler_message["message"]["level"],
                        "message": compiler_message["message"]["message"],
                        "spans": compiler_message["message"]["spans"],
                    })
            except Exception as e:
                # TODO: 解析错误，详细错误处理
                raise e
        # compile_errors = deduplicate_by_key(compile_errors, lambda x: x["rendered"])
        return {
            "success": len(compile_errors) == 0,
            "output": cargo_check_stderr,
            "errors": compile_errors,
        }
    except Exception as e:
        # TODO: 解析错误，详细错误处理
        raise e


def cargo_bench(project_dir, bench_name: str, ignore_codes: Optional[list[str]] = None, no_run: bool = False) -> Dict[str, Any]:
    """运行 cargo bench 命令进行基准测试"""
    ignore_codes = ignore_codes or []
    cargo_bench_command = [CARGO_BIN, "bench", "--bench", bench_name, "--message-format", "json"]
    if no_run:
        cargo_bench_command.append("--no-run")
    try:
        cargo_bench_process = subprocess.run(
            cargo_bench_command,
            capture_output=True,
            text=True,
            cwd=os.path.abspath(project_dir)
        )
        cargo_bench_stderr = cargo_bench_process.stderr
        output_lines = cargo_bench_process.stdout.split("\n")
        compile_errors = []
        for output_line in output_lines:
            if output_line.strip() == "":
                continue
            try:
                cargo_check_output = json.loads(output_line)
                if cargo_check_output["reason"] == "compiler-message":
                    compiler_message = cargo_check_output
                    if compiler_message["message"]["level"] != "error":
                        continue
                    if compiler_message["message"]["code"] and compiler_message["message"]["code"]["code"] in ignore_codes:
                        continue
                    if "aborting due to" in compiler_message["message"]["message"]:
                        continue
                    # TODO：这里目前忽略模块信息，仅针对单模块进行检查
                    compile_errors.append({
                        "rendered": compiler_message["message"]["rendered"],
                        "message_type": compiler_message["message"]["$message_type"],
                        "children": compiler_message["message"]["children"],
                        "code": compiler_message["message"]["code"],
                        "level": compiler_message["message"]["level"],
                        "message": compiler_message["message"]["message"],
                        "spans": compiler_message["message"]["spans"],
                    })
            except Exception as e:
                # TODO: 这里可能是解析错误，也可能是 bench 的输出
                continue
        return {
            "success": len(compile_errors) == 0,
            "output": cargo_bench_stderr,
            "errors": compile_errors,
        }
    except Exception as e:
        raise e

def extract_error_output(test_case_name, cargo_test_stdout):
    """
    从输出中提取指定测试用例的错误信息
    通过测试名称（如 test_max_heap）查找并提取该测试的错误输出直到空行
    """
    # 捕获所有在 '---- test_case_name stdout ----' 后的内容，直到下一个空行或下一个测试的输出
    pattern = rf"----\s*(?:tests::)?{test_case_name}\s*stdout\s*----\s*(.*?)\s*(?=----|\Z)"

    # 使用 re.DOTALL 让 '.' 匹配换行符
    match = re.search(pattern, cargo_test_stdout, re.DOTALL)

    if match:
        return match.group(1).strip()

    return ""

def cargo_test(project_path, test_name, timeout=120):
    """
    执行测试，并在超时后返回已产生的输出。
    使用 selectors 进行非阻塞的 I/O 读取。
    """
    # cur_cwd = os.getcwd()
    # os.chdir(project_path)
    env = os.environ.copy()
    env['RUST_BACKTRACE'] = '1'
    cargo_check_command = [CARGO_BIN, "test", "--test", test_name, "--message-format", "json"]

    # 初始化输出缓冲区
    cargo_test_stdout = []
    cargo_test_stderr = []

    # 创建 selectors 对象
    sel = selectors.DefaultSelector()

    try:
        # 启动子进程
        process = subprocess.Popen(
            cargo_check_command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,  # 行缓冲
            cwd=os.path.abspath(project_path)
        )

        # 注册 stdout 和 stderr 到 selectors
        sel.register(process.stdout, selectors.EVENT_READ)
        sel.register(process.stderr, selectors.EVENT_READ)

        start_time = time.time()
        timeout_occurred = False

        while True:
            elapsed_time = time.time() - start_time
            remaining_time = timeout - elapsed_time
            if remaining_time <= 0:
                timeout_occurred = True
                break

            events = sel.select(timeout=remaining_time)

            for key, _ in events:
                data = key.fileobj.readline()
                if data:
                    if key.fileobj is process.stdout:
                        cargo_test_stdout.append(data)
                    elif key.fileobj is process.stderr:
                        cargo_test_stderr.append(data)
                else:
                    # EOF
                    sel.unregister(key.fileobj)

            # 检查子进程是否已经结束
            if process.poll() is not None:
                break

        if timeout_occurred:
            # 超时，终止子进程
            process.kill()
            # 继续读取剩余的数据，避免阻塞
            while True:
                events = sel.select(timeout=1)
                if not events:
                    break
                for key, _ in events:
                    data = key.fileobj.readline()
                    if data:
                        if key.fileobj is process.stdout:
                            cargo_test_stdout.append(data)
                        elif key.fileobj is process.stderr:
                            cargo_test_stderr.append(data)
                    else:
                        sel.unregister(key.fileobj)
            filtered_stdout = [line for line in cargo_test_stdout if
                               not (line and (line[0] == "{" or line[0] == "[")) or not json.loads(line)]
            return {
                "success": True,
                "timeout": True,
                "output": "".join(filtered_stdout),
                "errors": [{"message": "Timeout expired during cargo test"}],
            }

        # 读取剩余的数据
        while True:
            events = sel.select(timeout=0)
            if not events:
                break
            for key, _ in events:
                data = key.fileobj.readline()
                if data:
                    if key.fileobj is process.stdout:
                        cargo_test_stdout.append(data)
                    elif key.fileobj is process.stderr:
                        cargo_test_stderr.append(data)
                else:
                    sel.unregister(key.fileobj)

        sel.close()

        # 合并输出
        cargo_test_stdout_str = "".join(cargo_test_stdout)
        cargo_test_stderr_str = "".join(cargo_test_stderr)

        # 检查子进程的返回码
        if process.returncode != 0:
            # 如果因为超时导致的非零退出码已经在前面处理
            pass

        # 正常处理输出
        output_lines = cargo_test_stdout_str.split("\n")
        compile_errors = []
        test_output_lines = []
        test_failed_cases = []
        build_result = False

        for output_line in output_lines:
            if output_line.strip() == "":
                continue
            try:
                cargo_test_output = json.loads(output_line)
                if cargo_test_output["reason"] == "compiler-message":
                    compiler_message = cargo_test_output
                    if compiler_message["message"]["level"] != "error":
                        continue
                    compile_errors.append({
                        "rendered": compiler_message["message"]["rendered"],
                        "message_type": compiler_message["message"]["$message_type"],
                        "children": compiler_message["message"]["children"],
                        "code": compiler_message["message"]["code"],
                        "level": compiler_message["message"]["level"],
                        "message": compiler_message["message"]["message"],
                        "spans": compiler_message["message"]["spans"],
                    })
                if cargo_test_output["reason"] == "build-finished":
                    build_result = cargo_test_output["success"]
            except json.JSONDecodeError:
                # 如果json解析出错，那么该行数据并非json数据
                test_output_lines.append(output_line)
                # 解析出成功和失败的测试用例
                pattern = r"test\s+(?:tests::)?(\w+)\s+\.\.\.\s+(\w+)"
                match = re.match(pattern, output_line)
                if match:
                    test_case_name = match.group(1)
                    result = match.group(2)
                    if result == "FAILED":
                        error_info = extract_error_output(test_case_name, cargo_test_stdout_str)
                        test_failed_cases.append({
                            "test_case_name": test_case_name,
                            "result": result,
                            "error_info": error_info
                        })

        if build_result:
            return {
                "success": True,
                "timeout": False,
                "output": "\n".join(test_output_lines),
                "errors": test_failed_cases
            }
        else:
            return {
                "success": len(compile_errors) == 0,
                "timeout": False,
                "output": cargo_test_stderr_str,
                "errors": compile_errors,
            }

    except Exception as e:
        raise e
    # finally:
    #     os.chdir(cur_cwd)