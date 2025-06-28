import difflib
import re
from collections import defaultdict
from typing import Optional, Dict
import difflib

CODE_BLOCK_CHANGE_PATTERN = re.compile(r"```(\w+):([^:]+):(\d+):(\d+)\n([\s\S]*?)```", re.DOTALL)

# TODO 根据prompt的格式修改该正则表达式
CODE_BLOCK_REPLACE_PATTERN = re.compile(
    r"```(\w+):([^:\n]+)\n<<<<<<< SEARCH\n([\s\S]*?)=======\n([\s\S]*?)>>>>>>> REPLACE\n```",
    re.DOTALL)


def extract_code_block_change_info(message: str) -> dict:
    return extract_code_block_change_info_with_replace(message)


def apply_changes(content: str, changes: list[dict]) -> str:
    return apply_changes_with_replace(content, changes)


def extract_code_block_change_info_with_line(message: str) -> dict:
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


def apply_changes_with_line(content: str, changes: list[dict]) -> str:
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
        if start_line < 0 or start_line > end_line:
            raise ValueError(f"Invalid line range: start_line={start_line}, end_line={end_line}")

        # 将新内容按行拆分
        new_lines = new_content.splitlines()

        # 计算行数差异（新内容的行数 - 旧内容的行数）
        old_line_count = end_line - start_line + 1
        new_line_count = len(new_lines)
        line_delta = new_line_count - old_line_count

        # 替换原始内容中的指定行
        lines[start_line: end_line + 1] = new_lines

        # 更新行号偏移量
        line_offset += line_delta

    # 将修改后的行重新合并为字符串
    return '\n'.join(lines)


# def extract_code_block_change_info_with_replace(message: str) -> dict:
#     """从消息中提取所有代码替换块"""
#     matches = CODE_BLOCK_REPLACE_PATTERN.findall(message)
#     replacements = defaultdict(list)
#     for match in matches:
#         lang, filepath, search, replace = match
#         replacements[filepath].append({
#             "lang": lang,
#             "filepath": filepath,
#             "search": search,
#             "replace": replace
#         })
#     return dict(replacements)


# def process_code_replacements(original_files: Dict[str, str], message: str) -> Dict[str, str]:
#     """处理所有文件的替换操作"""
#     all_replacements = extract_code_block_replace_info(message)
#     results = {}
#
#     for filepath, replacements in all_replacements.items():
#         content = original_files.get(filepath, "")
#         results[filepath] = apply_replacements(content, replacements)
#
#     # 处理创建新文件的情况
#     for filepath, replacements in all_replacements.items():
#         if filepath not in original_files:
#             if any(not repl["search"] and repl["replace"] for repl in replacements):
#                 results[filepath] = "\n".join([repl["replace"] for repl in replacements if repl["replace"]])
#
#     return results

def extract_code_block_change_info_with_replace(message: str) -> dict:
    """
    从消息中提取基于 SEARCH/REPLACE 代码块的变更信息。

    代码块格式示例：
    ```lang:filepath
    <<<<<<< SEARCH
    [待替换的原始内容]
    =======
    [用于替换的新内容]
    >>>>>>> REPLACE
    ```

    这里修正了正则表达式：
      - 移除了行首^的限制；
      - 在各行结束处允许可选空白字符。

    返回示例：
    {
        "file.rs": [
            {
                "language": "rust",
                "filepath": "file.rs",
                "search": "原始内容...",
                "replace": "替换的新内容..."
            },
            ...
        ]
    }
    """
    # pattern = re.compile(r"```(\w+):([^:\n]+)\n<<<<<<< SEARCH\n([\s\S]*?)=======\n([\s\S]*?)>>>>>>> REPLACE\n```",
    #                      re.DOTALL)



    results = CODE_BLOCK_REPLACE_PATTERN.findall(message)
    changes = defaultdict(list)
    for result in results:
        language = result[0]
        filepath = result[1]
        search_content = result[2].strip("\n")
        replace_content = result[3].strip("\n")
        # 去除可能的行号
        new_search_content_lines = []
        search_content_lines = search_content.splitlines()
        if len(search_content_lines) == 1:
            # 当生成的搜索代码块只有一行时，为避免缩进问题，去除行首空格
            new_search_content_lines.append(search_content_lines[0].lstrip(" "))
        else:
            for search_content_line in search_content_lines:
                stripped_search_content_line = search_content_line.lstrip(" ")
                if ":" in stripped_search_content_line:
                    line_number, actual_search_content_line = stripped_search_content_line.split(":", 1)
                    if line_number.isdigit():
                        new_search_content_lines.append(actual_search_content_line)
                        continue
                    else:
                        new_search_content_lines.append(search_content_line)
                else:
                    new_search_content_lines.append(search_content_line)
        search_content = "\n".join(new_search_content_lines)
        new_replace_content_lines = []
        for replace_content_line in replace_content.splitlines():
            stripped_replace_content_line = replace_content_line.lstrip(" ")
            if ":" in stripped_replace_content_line:
                line_number, actual_replace_content_line = stripped_replace_content_line.split(":", 1)
                if line_number.isdigit():
                    new_replace_content_lines.append(actual_replace_content_line)
                    continue
                else:
                    new_replace_content_lines.append(replace_content_line)
            else:
                new_replace_content_lines.append(replace_content_line)
        replace_content = "\n".join(new_replace_content_lines)
        changes[filepath].append({
            "language": language,
            "filepath": filepath,
            "search": search_content,
            "replace": replace_content
        })
    return dict(changes)


def apply_changes_with_replace(content: str, changes: list[dict]) -> str:
    """
    将基于 SEARCH/REPLACE 代码块提取的变更应用到原始内容中。

    参数:
        content (str): 原始内容
        changes (list[dict]): 多个修改项，每项包含：
            - "search" (str): 待查找的原始内容（必须逐字符匹配，包括所有空格、缩进等）
            - "replace" (str): 用于替换的新内容

    对每个变更，仅替换首次匹配到的内容。如果在内容中找不到待替换的文本，则抛出异常。

    返回:
        str: 应用所有修改后的内容
    """
    for change in changes:
        search_text = change["search"]
        replace_text = change["replace"]

        index = content.find(search_text)
        if index == -1:
            raise ValueError(f"无法在内容中找到需要替换的文本：\n{search_text}")

        # 替换首次匹配到的内容
        content = content[:index] + replace_text + content[index + len(search_text):]
    return content


def test_line():
    # 创建新文件
    create_file_without_content_message = """
```rust:file.rs:1:1
```
        """
    create_file_with_content_message = """
```rust:file.rs:1:1
fn main() {
println!("Hello, world!");
}
```
        """
    insert_file_message = """
```rust:file.rs:4:4
fn greet() {
println!("Hello, world!");
}
```
        """
    update_file_message = """
```rust:file.rs:5:5
println!("Hello, Greet!");
println!("Hello, world!");
```
        """
    delete_file_message = """
```rust:file.rs:6:6
```
        """
    content = ""
    for message in [create_file_without_content_message, create_file_with_content_message, insert_file_message,
                    update_file_message, delete_file_message]:
        changes = extract_code_block_change_info_with_line(message)
        print(changes)
        content = apply_changes_with_line(content, changes["file.rs"])
        print(content)
        print("=====================================")


def test_replace():
    # 创建新文件
    create_file_without_content_message = """
```rust:file.rs
<<<<<<< SEARCH
=======
>>>>>>> REPLACE
```
    """
    create_file_with_content_message = """
```rust:file.rs
<<<<<<< SEARCH
=======
fn main() {
    println!("Hello, world!");
}
>>>>>>> REPLACE
```
    """
    insert_file_message = """
```rust:file.rs
<<<<<<< SEARCH
fn main() {
    println!("Hello, world!");
}
=======
fn main() {
    println!("Hello, world!");
}

fn greet() {
    println!("Hello, world!");
}
>>>>>>> REPLACE
```
    """
    update_file_message = """
```rust:file.rs
<<<<<<< SEARCH
    println!("Hello, world!");
=======
    println!("Hello, Greet!");
    println!("Hello, world!");
>>>>>>> REPLACE
```
    """
    delete_file_message = """
```rust:file.rs
<<<<<<< SEARCH
    println!("Hello, world!");
=======
>>>>>>> REPLACE
```
    """

    content = ""
    for message in [
        create_file_without_content_message,
        create_file_with_content_message,
        insert_file_message,
        update_file_message,
        delete_file_message
    ]:
        changes = extract_code_block_change_info_with_replace(message)
        print(changes)
        content = apply_changes_with_replace(content, changes["file.rs"])
        print(content)
        print("=====================================")


if __name__ == '__main__':
    # test_line()
    test_replace()
