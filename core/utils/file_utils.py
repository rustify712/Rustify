import os
from typing import Annotated, List, TypedDict, Literal, Optional, Callable


class ListFiles(TypedDict):
    type: Literal["dir", "file"]
    path: str
    children: List["ListFiles"]

def list_files_flat(
    dirpath: str,
    ignore: Optional[List[str]] = None,
    ignore_func: Optional[Callable] = None,
    content: Optional[bool] = False
) -> List[dict]:
    """递归列出目录下的全部文件。

    Args:
        dirpath (str): 目录路径。
        ignore: (Optional[List[str]]) 忽略的文件或目录。
        ignore_func: (Optional[Callable]) 忽略的文件或目录的函数。

    Returns:
        list[str]: 目录下的文件列表。
    """
    file_list = []
    for root, dirs, files in os.walk(dirpath):
        for entry in files:
            path = os.path.join(root, entry)
            if ignore and path in ignore:
                continue
            if ignore_func and ignore_func(path):
                continue
            current_file = {
                "type": "file",
                "path": os.path.relpath(path, dirpath),
                "abspath": path
            }
            if content:
                with open(path, "r", encoding="utf-8") as f:
                    current_file["content"] = f.read()
            file_list.append(current_file)
    return file_list

def list_files(
    dirpath: str,
    ignore: Optional[List[str]] = None,
    ignore_func: Optional[Callable] = None
) -> ListFiles:
    """递归列出目录下的全部文件。

    Args:
        dirpath (str): 目录路径。
        ignore: (Optional[List[str]]) 忽略的文件或目录。
        ignore_func: (Optional[Callable]) 忽略的文件或目录的函数。

    Returns:
        list[str]: 目录下的文件列表。
    """
    if not os.path.isdir(dirpath):
        raise FileNotFoundError(f"目录 '{dirpath}' 不存在。")

    def inner_list_files(dirpath: str):
        file_list = []
        for entry in os.listdir(dirpath):
            path = os.path.join(dirpath, entry)
            if ignore and path in ignore:
                continue
            if ignore_func and ignore_func(path):
                continue
            current_file = {
                "path": os.path.relpath(path, dirpath),
                "children": []
            }
            if os.path.isdir(path):
                current_file["type"] = "dir"
                current_file["children"] = inner_list_files(path)  # 递归调用，增加缩进
            else:
                current_file["type"] = "file"
            file_list.append(current_file)
        return file_list

    return {
        "type": "dir",
        "path": dirpath,
        "children": inner_list_files(dirpath)
    }


def pretty_files(file: dict):
    files_str = ""

    def inner_pretty_files(file: dict, indent: int = 0):
        nonlocal files_str
        files_str += " " * indent + f"[{file['type'].upper()}] {file['path']}\n"
        if file["type"] == "dir":
            for child in file["children"]:
                inner_pretty_files(child, indent + 2)

    inner_pretty_files(file)
    return files_str

def pretty_files_with_summary(file: dict, file_summary_dict):
    files_str = ""

    def inner_pretty_files(file: dict, indent: int = 0, parent_path: str = ""):
        nonlocal files_str
        file_summary = file_summary_dict.get(file["path"], "")
        files_str += " " * indent + f"[{file['type'].upper()}] {file['path']}: {file_summary}\n"
        if file["type"] == "dir":
            for child in file["children"]:
                inner_pretty_files(child, indent + 2, f"{parent_path}/{file['path']}".lstrip("/"))

    inner_pretty_files(file)
    return files_str

def add_line_numbers(content: str):
    """为内容添加行号。"""
    lines = content.split("\n")
    line_digit_count = len(str(len(lines)))
    lines_with_line_numbers = [f"{i + 1:>{line_digit_count}}:{line}" for i, line in enumerate(lines)]
    return "\n".join(lines_with_line_numbers)