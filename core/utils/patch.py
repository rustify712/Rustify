import difflib
import re
from collections import defaultdict
from typing import Optional

CODE_BLOCK_CHANGE_PATTERN = re.compile(r"```(\w+):([^:]+):(\d+):(\d+)\n([\s\S]*?)```", re.DOTALL)


def extract_code_block_change_info(message: str) -> dict:
    """从消息中提取代码块的变更信息"""
    results = CODE_BLOCK_CHANGE_PATTERN.findall(message)
    changes = defaultdict(list)
    if len(results) > 0:
        for result in results:
            language = result[0]
            filepath = result[1]
            start_line = int(result[2])
            end_line = int(result[3])
            new_content = result[4]
            changes[filepath].append({
                "language": language,
                "filepath": filepath,
                "start_line": start_line,
                "end_line": end_line,
                "content": new_content
            })
    return dict(changes)


def apply_changes(content: str, changes: list[dict]) -> str:
    """应用代码块变更到原始内容

    Args:
        content (str): 原始内容
        changes (list[dict]): 包含多个修改的列表，每个修改是一个字典，包含以下键：
            - "start_line" (int): 修改的起始行号（从 1 开始）
            - "end_line" (int): 修改的结束行号（从 1 开始）
            - "content" (str): 替换的新内容

    Returns:
        str: 应用所有修改后的内容
    """
    # 将原始内容按行拆分为列表
    lines = content.splitlines()
    changes.sort(key=lambda x: x["start_line"])

    # 初始化行号偏移量
    line_offset = 0

    # 对每个修改进行处理
    for change in changes:
        # 调整行号，考虑前面的修改对行号的影响
        start_line = max(change["start_line"] + line_offset - 1, 0)
        end_line = change["end_line"] + line_offset - 1
        new_content = change["content"]

        # 确保行号在有效范围内
        if start_line < 0 or end_line >= len(lines) or start_line > end_line:
            raise ValueError(f"Invalid line range: start_line={start_line}, end_line={end_line}")

        # 将新内容按行拆分
        new_lines = new_content.splitlines()

        # 计算行数差异（新内容的行数 - 旧内容的行数）
        old_line_count = end_line - start_line + 1
        new_line_count = len(new_lines)
        line_delta = new_line_count - old_line_count

        # 替换原始内容中的指定行
        lines[start_line : end_line + 1] = new_lines

        # 更新行号偏移量
        line_offset += line_delta

    # 将修改后的行重新合并为字符串
    return '\n'.join(lines)