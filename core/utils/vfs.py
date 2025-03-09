import os
import time
import threading
from abc import abstractmethod
from contextlib import contextmanager
from typing import Generic, Optional, TypeVar
from collections import UserDict


class VirtualFile:

    @abstractmethod
    def read(self, size: int = -1) -> str:
        """读取文件内容"""
        raise NotImplementedError()

    @abstractmethod
    def write(self, content: str) -> int:
        """写入文件内容"""
        raise NotImplementedError()

    @abstractmethod
    def seek(self, offset: int, whence: int = 0):
        """移动文件指针"""
        raise NotImplementedError()

    @abstractmethod
    def tell(self) -> int:
        """获取文件指针位置"""
        raise NotImplementedError()

    @abstractmethod
    def close(self):
        """关闭文件"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def content(self):
        """文件内容"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def closed(self):
        """判断文件是否已关闭"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def modified(self):
        """判断文件是否已修改"""
        raise NotImplementedError()

class VirtualFileSystem:

    @abstractmethod
    def init(self, *args, **kwargs):
        """初始化文件系统"""
        raise NotImplementedError

    @abstractmethod
    def exists(self, filepath: str) -> bool:
        """判断文件是否存在"""
        raise NotImplementedError

    @abstractmethod
    @contextmanager
    def open(self, filepath: str, mode: str = "r") -> VirtualFile:
        """打开文件"""
        raise NotImplementedError()

    @abstractmethod
    def remove(self, filepath: str):
        """删除文件"""
        raise NotImplementedError()

    @abstractmethod
    def list_files(self, dirpath: str) -> list[str]:
        """列出目录下的文件"""
        raise NotImplementedError()

class ReadWriteLock:
    """读写锁

    默认支持写优先
    """

    def __init__(self, write_priority: bool = True):
        self._write_priority = write_priority
        self._lock = threading.Condition(threading.Lock())
        self._reading = False  # 是否有读线程正在持有读锁
        self._reader_count = 0  # 读锁计数
        self._writing = False  # 是否有写线程正在持有写锁
        self._writer_waiting_count = 0  # 等待写锁的线程数

    def _can_read(self) -> bool:
        """判断是否可以读
        1. 无线程持有写锁
        2. 写优先时，无线程等待写锁
        """
        if self._write_priority:
            return not (self._writing or self._writer_waiting_count > 0)
        return not self._writing

    def _can_write(self) -> bool:
        """判断是否可以写
        1. 无线程持有读锁
        2. 无线程持有写锁
        """
        return not (self._reader_count > 0 or self._writing)

    def _acquire_read(self, timeout: Optional[float] = None) -> bool:
        """获取读锁

        Args:
            timeout: 超时时间（秒）, None 表示一直等待

        Returns:
            bool: 是否获取到锁
        """
        deadline = None if timeout is None else time.time() + timeout

        with self._lock:
            while not self._can_read():
                if deadline is not None:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        return False
                    if not self._lock.wait(remaining):
                        return False
                else:
                    self._lock.wait()
            self._reader_count += 1
            return True

    def _release_read(self):
        """释放读锁"""
        with self._lock:
            self._reader_count -= 1
            if self._reader_count == 0:
                self._lock.notify_all()

    def _acquire_write(self, timeout: Optional[float] = None) -> bool:
        """获取写锁

        Args:
            timeout: 超时时间（秒）, None 表示一直等待

        Returns:
            bool: 是否获取到锁
        """
        deadline = None if timeout is None else time.time() + timeout

        with self._lock:
            self._writer_waiting_count += 1
            try:
                while not self._can_write():
                    if deadline is not None:
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            return False
                        if not self._lock.wait(remaining):
                            return False
                    else:
                        self._lock.wait()
                self._writing = True
                return True
            finally:
                self._writer_waiting_count -= 1

    def _release_write(self):
        """释放写锁"""
        with self._lock:
            self._writing = False
            self._lock.notify_all()

    @contextmanager
    def read_lock(self, timeout: Optional[float] = None):
        """读锁上下文管理器

        Args:
            timeout: 超时时间（秒）, None 表示一直等待

        Raises:
            TimeoutError: 获取读锁超时
        """
        if not self._acquire_read(timeout):
            raise TimeoutError("acquire read lock timeout")
        try:
            yield
        finally:
            self._release_read()

    @contextmanager
    def write_lock(self, timeout: Optional[float] = None):
        """写锁上下文管理器

        Args:
            timeout: 超时时间（秒）, None 表示一直等待

        Raises:
            TimeoutError: 获取写锁超时
        """
        if not self._acquire_write(timeout):
            raise TimeoutError("acquire write lock timeout")
        try:
            yield
        finally:
            self._release_write()

class MemoryFile(VirtualFile):

    def __init__(self, content: str = "", mode: str = ""):
        self._content = content
        self._pointer = 0
        self._mode = mode
        self._closed = False
        self._modified = False
        self._lock = ReadWriteLock(write_priority=True)
        self._created_time = time.time()
        self._modified_time = self._created_time

    def read(self, size: int = -1) -> str:
        """读取文件内容

        Args:
            size: 读取的字符数，-1 表示读取全部内容

        Returns:
            str: 读取的内容

        Raises:
            ValueError: 文件已关闭
        """
        with self._lock.read_lock():
            if self._closed:
                raise ValueError("I/O operation on closed file")
            if "r" not in self._mode:
                raise IOError("File not open for reading")

            if size < 0:
                content = self._content[self._pointer:]
                self._pointer = len(self._content)
            else:
                content = self._content[self._pointer:self._pointer + size]
                self._pointer += size
            return content

    def write(self, content: str) -> int:
        """写入文件内容

        a 模式：从文件末尾开始写入
        w 模式：从当前位置开始写入，覆盖原有内容
        """
        with self._lock.write_lock():
            if self._closed:
                raise ValueError("I/O operation on closed file")
            if "w" not in self._mode and "a" not in self._mode:
                raise IOError("File not open for writing")

            if self._mode.startswith("a"):
                # a 模式：从文件末尾开始写入
                self._pointer = len(self._content)
                self._content += content
            else:
                # w 模式：从当前位置开始写入，覆盖原有内容
                self._content = self._content[:self._pointer] + content

            self._pointer += len(content)
            self._modified = True
            self._modified_time = time.time()
            return len(content)

    def seek(self, offset: int, whence: int = 0):
        """移动文件指针

        Args:
            offset: 偏移量
            whence: 偏移起始位置，0: 文件起始位置，1: 当前位置，2: 文件末尾位置
        """
        with self._lock.read_lock():
            if self._closed:
                raise ValueError("I/O operation on closed file")

            if whence == 0:
                self._pointer = offset
            elif whence == 1:
                self._pointer += offset
            elif whence == 2:
                self._pointer = len(self._content) + offset
            else:
                raise ValueError("Invalid whence")

            self._pointer = max(0, min(self._pointer, len(self._content)))

    def tell(self) -> int:
        """获取文件指针位置"""
        with self._lock.read_lock():
            if self._closed:
                raise ValueError("I/O operation on closed file")
            return self._pointer

    def close(self):
        """关闭文件"""
        with self._lock.write_lock():
            self._closed = True

    @property
    def closed(self):
        """判断文件是否已关闭"""
        return self._closed

    @property
    def content(self):
        return self._content

    @property
    def modified(self):
        return self._modified


KT = TypeVar("KT")
VT = TypeVar("VT")
class SafeDict(UserDict, Generic[KT, VT]):

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()

    def __setitem__(self, key: KT, value: VT):
        with self._lock:
            super().__setitem__(key, value)

    def __delitem__(self, key: KT):
        with self._lock:
            super().__delitem__(key)

    def __getitem__(self, key: KT) -> VT:
        with self._lock:
            return super().__getitem__(key)

class MemoryFileSystem(VirtualFileSystem):

    def __init__(self):
        self._files: SafeDict[str, MemoryFile] = SafeDict()
        self._root_dir = "/"

    @classmethod
    def init(cls, root_dir: str):
        fs = MemoryFileSystem()
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"No such directory: '{root_dir}'")
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                filepath = os.path.join(root, file)
                with open(filepath, "r") as f:
                    content = f.read()
                fs._files[filepath] = MemoryFile(content, "r")
        fs._root_dir = root_dir
        return fs


    @classmethod
    def _normalize_path(cls, path: str) -> str:
        filepath = "/" + path.lstrip("/").replace("\\", "/")
        # 处理路径中的 "." 和 ".."
        parts = []
        for part in filepath.split("/"):
            if not part or part == ".":
                continue
            if part == "..":
                if parts:
                    parts.pop()
            else:
                parts.append(part)
        return "/" + "/".join(parts)

    def exists(self, filepath: str) -> bool:
        filepath = self._normalize_path(filepath)
        return filepath in self._files

    @contextmanager
    def open(self, filepath: str, mode: str = "r") -> MemoryFile:
        filepath = self._normalize_path(filepath)
        if "r" in mode and not self.exists(filepath):
            raise FileNotFoundError(f"No such file: '{filepath}'")

        try:
            if mode.startswith("r"):
                file = self._files[filepath]
                if file.closed:
                    file = MemoryFile(file.content, mode)
                    self._files[filepath] = file
            elif mode.startswith("w"):
                file = MemoryFile("", mode)
                self._files[filepath] = file
            elif mode.startswith("a"):
                if self.exists(filepath):
                    file = self._files[filepath]
                    file = MemoryFile(file.content, mode)
                else:
                    file = MemoryFile("", mode)
                self._files[filepath] = file
            else:
                raise ValueError(f"Invalid mode: '{mode}'")

            try:
                yield file
            finally:
                if not file.closed:
                    file.close()

        except Exception as e:
            raise e

    def remove(self, filepath: str):
        filepath = self._normalize_path(filepath)
        if self.exists(filepath):
            del self._files[filepath]

    def list_files(self, dirpath: str, recursive: bool = False) -> list[str]:
        """列出目录下的文件

        Args:
            dirpath: 目录路径
            recursive: 是否递归列出子目录下的文件
        """
        dirpath = self._normalize_path(dirpath)
        if not dirpath.endswith("/"):
            dirpath += "/"

        files = set()
        for filepath, file in self._files.items():
            if not filepath.startswith(dirpath):
                continue

            relpath = filepath[len(dirpath):]
            if recursive:
                files.add(relpath)
            else:
                parts = relpath.split("/")
                if len(parts) == 1:
                    files.add(relpath)

        return sorted(list(files))

    def sync_to_disk(self):
        """同步内存文件系统到磁盘"""
        for filepath, file in self._files.items():
            if file.modified:
                disk_path = f"disk{filepath}"