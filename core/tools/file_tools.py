import os
from typing import Annotated, List, Optional, Tuple
from core.utils.file_utils import list_files


def create_file_tool(
        filepath: Annotated[str, "文件路径"],
        content: Annotated[str, "文件内容"]
) -> None:
    """创建文件。

    Args:
        filepath (str): 要创建的文件路径。
        content (str): 文件内容。

    Raises:
        FileExistsError: 如果文件已存在。
        IOError: 如果无法创建文件。
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.exists(filepath):
        raise FileExistsError(f"文件 '{filepath}' 已存在。")
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        raise IOError(f"创建文件时发生错误: {e}")


def read_file_tool(
        filepath: Annotated[str, "要读取的文件的路径"],
        line_numbers: Annotated[Optional[List[int]], "指定要读取的行号列表（从1开始）。如果为 None，则读取所有行。"] = None,
        show_line_number: Annotated[Optional[bool], "是否显示行号。默认为 True。"] = True
) -> Annotated[str, "读取的文件内容。当 show_line_number 为 True 时，返回的内容会包含行号。"]:
    """读取文件内容, 支持读取指定行号的内容。"""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"文件 '{filepath}' 不存在。")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) == 0:
                return "文件内容为空。"
            if line_numbers is None:
                result = lines
            else:
                max_line_num = len(lines)
                result = []
                for number in line_numbers:
                    if 1 <= number <= max_line_num:
                        result.append(lines[number - 1])
                    else:
                        raise IndexError(f"行号 {number} 超出文件范围（1-{max_line_num}）。")
            if show_line_number:
                line_digit_count = len(str(len(result)))
                result = [f"{i + 1:>{line_digit_count}}:{line}" for i, line in enumerate(result)]
            return "".join(result)
    except Exception as e:
        raise IOError(f"读取文件时发生错误: {e}")


def modify_file_tool(
        filepath: Annotated[str, "要修改的文件路径"],
        contents_with_line_number: Annotated[List[Tuple[int, str]], "修改后的内容列表, 格式为 [(行号, 内容)]"],
) -> None:
    """修改文件指定行号的内容。"""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"文件 '{filepath}' 不存在。")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        max_line_num = len(lines)
        for number, content in contents_with_line_number:
            if 1 <= number <= max_line_num:
                lines[number - 1] = content
            else:
                raise IndexError(f"行号 {number} 超出文件范围（1-{max_line_num}）。")
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception as e:
        raise IOError(f"修改文件时发生错误: {e}")


def file_insert_content_tool(
    filepath: Annotated[str, "待插入内容的文件路径"],
    content: Annotated[str, "待插入的内容"],
    line_number: Annotated[Optional[int], "插入的行号，从 1 开始，默认为 None，表示在文件末尾插入。"] = None
) -> None:
    """在指定行号后插入内容, 如果行号为 None，则在文件末尾插入。"""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"文件 '{filepath}' 不存在。")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if line_number is None or len(lines) == 0:
            lines.append(content)
        else:
            max_line_num = len(lines)
            if 1 <= line_number <= max_line_num:
                lines.insert(line_number, content)
            else:
                raise IndexError(f"行号 {line_number} 超出文件范围（1-{max_line_num}）。")
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception as e:
        raise IOError(f"插入内容时发生错误: {e}")


def file_append_content_tool(
        filepath: Annotated[str, "文件路径"],
        content: Annotated[str, "要追加的内容"]
) -> None:
    """在文件末尾追加内容。

    Warnings: 追加内容时，不会在内容前添加换行符，如需换行请在 content 中添加。

    Args:
        filepath (str): 要追加内容的文件路径。
        content (str): 要追加的内容。

    Raises:
        FileNotFoundError: 如果文件不存在。
        IOError: 如果无法修改文件。
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"文件 '{filepath}' 不存在。")
    try:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        raise IOError(f"追加内容时发生错误: {e}")


def file_delete_content_tool(
        filepath: Annotated[str, "文件路径"],
        line_numbers: Annotated[List[int], "要删除的行号列表"]
) -> None:
    """删除文件指定行号的内容。

    Args:
        filepath (str): 要删除内容的文件路径。
        line_numbers (List[int]): 要删除的行号列表。

    Raises:
        FileNotFoundError: 如果文件不存在。
        IndexError: 如果行号超出文件范围。
        IOError: 如果无法修改文件。
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"文件 '{filepath}' 不存在。")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        max_line_num = len(lines)
        for number in line_numbers:
            if 1 <= number <= max_line_num:
                del lines[number - 1]
            else:
                raise IndexError(f"行号 {number} 超出文件范围（1-{max_line_num}）。")
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception as e:
        raise IOError(f"删除内容时发生错误: {e}")


def delete_file_tool(
        filepath: Annotated[str, "文件路径"]
) -> None:
    """删除文件。

    Args:
        filepath (str): 要删除的文件路径。

    Raises:
        FileNotFoundError: 如果文件不存在。
        IOError: 如果无法删除文件。
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"文件 '{filepath}' 不存在。")
    try:
        os.remove(filepath)
    except Exception as e:
        raise IOError(f"删除文件时发生错误: {e}")


def list_files_tool(
    dirpath: Annotated[str, "目录路径"]
) -> Annotated[str, "目录下的文件列表"]:
    """递归列出目录下的全部文件。

    Args:
        dirpath (str): 目录路径。

    Returns:
        str
    """
    top_file = list_files(dirpath)
    files_str = ""

    def pretty_files(file: dict, indent: int = 0):
        nonlocal files_str
        files_str += " " * indent + f"[{file['type'].upper()}] {file['path']}\n"
        if file["type"] == "dir":
            for child in file["children"]:
                pretty_files(child, indent + 2)

    def pretty_files1(files: List[dict], indent: int = 0):
        nonlocal files_str
        for file in files:
            files_str += " " * indent + f"[{file['type'].upper()}] {file['path']}\n"
            if file["type"] == "dir":
                pretty_files1(file["children"], indent + 2)

    pretty_files(top_file)
    # pretty_files1(top_file["children"])
    return files_str
