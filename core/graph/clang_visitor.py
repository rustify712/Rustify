import os
from abc import abstractmethod
from typing import Generic, List, Optional, Tuple, TypeVar, Callable
import clang.cindex

from core.config import Config

try:
    if Config.LIB_CLANG:
        clang.cindex.Config.set_library_file(Config.LIB_CLANG)
except Exception as e:
    pass

Node = TypeVar("Node")


class NodeVisitor(Generic[Node]):
    """访问语法树节点的遍历器基类
    需要为特定的语言编写具体的访问函数
    """

    @abstractmethod
    def parse(self, path, unsaved_files=None, **kwargs):
        """解析代码生成语法树, 返回语法树根节点

        Args:
            path: str 代码文件路径
            unsaved_files: List[Tuple[str, str]] 未保存的文件内容，例如：[("a.c", "int main() { return 0; }")]
        """
        raise NotImplementedError

    @abstractmethod
    def node_type(self, node: Node) -> str:
        """获取节点类型

        Args:
            node: Node 语法树节点
        """
        raise NotImplementedError

    @abstractmethod
    def node_text(self, node: Node) -> str:
        """获取节点文本

        Args:
            node: Node 语法树节点
            encoding: str 编码

        Returns:
            str: 节点文本
        """
        raise NotImplementedError

    @abstractmethod
    def node_children(self, node: Node) -> List[Node]:
        """获取节点的子节点

        Args:
            node: Node 语法树节点

        Returns:
            List[Node]: 子节点列表
        """
        raise NotImplementedError

    def search_child_by(self, node: Node, func: Callable[[Node], bool]) -> Node:
        """查找子节点

        Args:
            node: Node 父节点
            func: Callable[[Node], bool] 比较函数

        Returns:
            Node 子节点
        """
        for child in self.node_children(node):
            if func(child):
                return child
        return None

    def search_by(self, node: Node, func: Callable[[Node], bool], deep: bool = False) -> Node:
        """递归查找子节点

        Args:
            node: Node 父节点
            func: Callable[[Node], bool] 比较函数
            deep: bool 是否深度优先搜索

        Returns:
            Node 子节点
        """

        def _search_by_with_dfs(_node: Node):
            """深度优先搜索"""
            if func(_node):
                return _node
            for child in self.node_children(_node):
                result = _search_by_with_dfs(child)
                if result:
                    return result
            return None

        def _search_by_with_bfs(_node: Node):
            """广度优先搜索"""
            queue = [node]
            while queue:
                current = queue.pop(0)
                if func(current):
                    return current
                queue.extend(self.node_children(current))
            return None

        if deep:
            return _search_by_with_dfs(node)
        else:
            return _search_by_with_bfs(node)

    def find_children_by(self, node: Node, func: Callable[[Node], bool]) -> List[Node]:
        """查找所有符合条件的子节点

        Args:
            node: Node 父节点
            func: Callable[[Node], bool] 比较函数

        Returns:
            List[Node]: 子节点
        """
        return [child for child in self.node_children(node) if func(child)]

    def find_by(self, node: Node, func: Callable[[Node], bool]) -> List[Node]:
        """递归查找所有符合条件的子节点

        Args:
            node: Node 父节点
            func: Callable[[Node, Node], bool] 比较函数

        Returns:
            Node: 子节点
        """
        node_list = []

        def _find_by(_node: Node):
            if func(_node):
                node_list.append(_node)
            for child in self.node_children(_node):
                _find_by(child)

        _find_by(node)
        return node_list

    def visit(self, node: Node):
        """根据语法树节点类型访问节点

        Args:
            node: Node 语法树节点
        """
        method = 'visit_' + self.node_type(node).lower()
        visit_func = getattr(self, method, self.generic_visit)
        return visit_func(node)

    def generic_visit(self, node: Node | List[Node]):
        """节点通用访问函数，如果节点没有明确的访问函数，则调用该函数。

        Args:
            node: Node | List[Node] 语法树节点
        """
        if isinstance(node, list):
            for item in node:
                self.visit(item)
        else:
            for child in self.node_children(node):
                self.visit(child)


class ClangNodeVisitor(NodeVisitor[clang.cindex.Cursor]):
    """C++语法树遍历器, 使用 clang 解析器

    """

    def __init__(self):
        self._tu = None
        self.parent_cursor: Optional[clang.cindex.Cursor] = None
        self._expected_files: Optional[str] = None
        self._file_content_map = {}

    def parse(self, path: str, args: List[str] = None, unsaved_files: List[Tuple[str, str]] = None, options: int = 1,
              **kwargs):
        """解析代码生成语法树

        Args:
            path: str 代码文件路径
            args: List[str] 编译参数列表
            unsaved_files: List[Tuple[str, str]] 未保存的文件内容，例如：[("a.c", "int main() { return 0; }")]
            options: int 解析选项，默认为 clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
                clang.cindex.TranslationUnit.PARSE_NONE: 无选项
                clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD: 使 AST 包含完整的预处理信息和宏定义
                clang.cindex.TranslationUnit.PARSE_INCOMPLETE: 允许不完整的代码（如仅有声明无定义的头文件）
                clang.cindex.TranslationUnit.PARSE_PRECOMPILED_PREAMBLE: 解析前将文件预编译并缓存，以加快后续解析速度
            kwargs: dict 其他参数
                expected_files: List[str] 期望解析的文件列表

        Returns:
            clang.cindex.Cursor: 语法树根节点光标
        """
        if unsaved_files:
            is_unsaved_file = False
            for unsaved_file in unsaved_files:
                if path == unsaved_file[0]:
                    is_unsaved_file = True
                    break
            if is_unsaved_file is None:
                raise ValueError("找不到源代码，请检查 unsaved_files")
        else:
            if not os.path.exists(path):
                raise FileNotFoundError(f"找不到文件: {path}")

        parent_dir = os.path.dirname(path)
        self._expected_files = kwargs.get("expected_files", None) or [
            os.path.join(parent_dir, filename) for filename in os.listdir(parent_dir) if
            os.path.isfile(os.path.join(parent_dir, filename))
        ] if parent_dir else []
        include_dirs = set([
            os.path.dirname(file)
            for file in self._expected_files
        ])
        include_args = [
            f"-I{include_dir}"
            for include_dir in include_dirs
        ]
        args = list(set(args + include_args))
        if path not in self._expected_files:
            self._expected_files.append(path)

        index = clang.cindex.Index.create()
        self._tu = index.parse(
            path=path,
            args=args,
            unsaved_files=unsaved_files,
            options=options
        )
        # 打印诊断信息
        for diag in self._tu.diagnostics:
            # 红色显示错误信息
            print(f"\033[1;31m[Diagnostic] {diag}\033[0m")
        return self._tu.cursor

    @staticmethod
    def from_language(language: str) -> "ClangNodeVisitor":
        """根据语言类型创建语法树遍历器

        Args:
            language: str 语言类型

        Returns:
            NodeVisitor: 语法树遍历器
        """
        if language == "c":
            return ClangCNodeVisitor()
        if language == "cpp":
            return ClangCPPNodeVisitor()
        raise ValueError(f"Unsupported language: {language}")

    def node_type(self, cursor: clang.cindex.Cursor) -> str:
        """获取节点类型

        Args:
            cursor: clang.cindex.Cursor 语法树节点
        """
        return cursor.kind.name

    # def node_text(self, cursor: clang.cindex.Cursor) -> str:
    #     """获取节点文本
    #
    #     Args:
    #         cursor: Node 语法树节点
    #
    #     Returns:
    #         str: 节点文本
    #     """
    #     tokens = list(cursor.get_tokens())
    #     if not tokens:
    #         start_offset = cursor.location.start.offset
    #         end_offset = cursor.location.end.offset
    #         return ""
    #
    #     result = []
    #     previous_token = None
    #
    #     # 限定拼接范围为当前 Cursor 的 extent 范围
    #     cursor_start = cursor.extent.start
    #     cursor_end = cursor.extent.end
    #
    #     for token in tokens:
    #         # 检查 token 是否在 Cursor 的有效范围内
    #         token_start = token.extent.start
    #         token_end = token.extent.end
    #
    #         if token_start.line < cursor_start.line or (
    #                 token_start.line == cursor_start.line and token_start.column < cursor_start.column
    #         ):
    #             # 当前 token 开始位置在 cursor 的开始位置之前，跳过
    #             continue
    #         if token_end.line > cursor_end.line or (
    #                 token_end.line == cursor_end.line and token_end.column > cursor_end.column
    #         ):
    #             # 当前 token 结束位置在 cursor 的结束位置之后，跳出循环
    #             break
    #
    #         if previous_token:
    #             prev_end = previous_token.extent.end
    #
    #             # 不在同一行时，添加换行符
    #             if token_start.line > prev_end.line:
    #                 result.append("\n" * (token_start.line - prev_end.line))
    #                 result.append(" " * (token_start.column - 1))
    #             elif token_start.column > prev_end.column:
    #                 # 在同一行，但不同列，填充空格
    #                 result.append(" " * (token_start.column - prev_end.column))
    #
    #         result.append(token.spelling)
    #         previous_token = token
    #
    #     return "".join(result)

    def node_text(self, cursor: clang.cindex.Cursor) -> str:
        """获取节点文本

        Args:
            cursor: Node 语法树节点

        Returns:
            str: 节点文本
        """
        start_location = cursor.extent.start
        end_location = cursor.extent.end

        if not start_location.file or not end_location.file:
            return ""

        if start_location.file.name not in self._file_content_map:
            with open(start_location.file.name, "r", encoding="utf-8", newline="\n") as f:
                content = f.read()
            self._file_content_map[start_location.file.name] = content
        else:
            content = self._file_content_map[start_location.file.name]

        start_offset = start_location.offset
        end_offset = end_location.offset

        return content[start_offset:end_offset]

    def node_children(self, cursor: clang.cindex.Cursor) -> List[clang.cindex.Cursor]:
        """获取节点的子节点

        Args:
            cursor: clang.cindex.Cursor 语法树节点

        Returns:
            List[clang.cindex.Cursor]: 子节点列表
        """
        return list(cursor.get_children())

    def visit(self, cursor: clang.cindex.Cursor):
        if cursor.location.file and cursor.location.file.name not in self._expected_files:
            # 去除不在期望文件列表中的节点
            return
        if not cursor.kind.is_translation_unit() and cursor.location.file is None:
            # 去除 clang 编译器生成的节点
            return
        # print(f"\033[1;32m{cursor.kind.name}, {cursor.spelling}\033[0m")
        # print(f"{self.node_text(cursor)}")
        return super().visit(cursor)

    def generic_visit(self, node: Node | List[Node]):
        """节点通用访问函数，如果节点没有明确的访问函数，则调用该函数。

        Args:
            node: Node | List[Node] 语法树节点
        """
        if isinstance(node, list):
            for item in node:
                self.visit(item)
        else:
            self.parent_cursor = node
            for child in self.node_children(node):
                self.visit(child)


class ClangCNodeVisitor(ClangNodeVisitor):

    def parse(self, path: str, args: List[str] = None, unsaved_files: List[Tuple[str, str]] = None, options: int = 1,
              **kwargs):
        """解析代码生成语法树，默认支持 C11 标准

        Args:
            path: str 代码文件路径
            args: List[str] 编译参数列表，默认值 ["-std=c11"]
            unsaved_files: List[Tuple[str, str]] 未保存的文件内容，例如：[("a.c", "int main() { return 0; }")]
            options: int 解析选项，默认为 clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
                clang.cindex.TranslationUnit.PARSE_NONE: 无选项
                clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD: 使 AST 包含完整的预处理信息和宏定义
                clang.cindex.TranslationUnit.PARSE_INCOMPLETE: 允许不完整的代码（如仅有声明无定义的头文件）
                clang.cindex.TranslationUnit.PARSE_PRECOMPILED_PREAMBLE: 解析前将文件预编译并缓存，以加快后续解析速度
            kwargs: dict 其他参数


        Returns:
            clang.cindex.Cursor: 语法树根节点光标
        """
        args = args or ["-std=c11"]
        return super().parse(path, args, unsaved_files, options, **kwargs)

    def visit_translation_unit(self, cursor: clang.cindex.Cursor):
        """访问语法树根节点"""
        self.generic_visit(cursor)

    def visit_preprocessing_directive(self, cursor: clang.cindex.Cursor):
        """访问预处理指令"""
        self.generic_visit(cursor)

    def visit_macro_definition(self, cursor: clang.cindex.Cursor):
        """访问宏定义
        示例：#define MAX 100
        """
        self.generic_visit(cursor)

    def visit_macro_expansion(self, cursor: clang.cindex.Cursor):
        """访问宏展开"""
        self.generic_visit(cursor)

    def visit_inclusion_directive(self, cursor: clang.cindex.Cursor):
        """访问包含指令
        示例：#include <stdio.h>
        """
        self.generic_visit(cursor)

    def visit_struct_decl(self, cursor: clang.cindex.Cursor):
        """访问结构体声明
        示例：
            struct Point {
                int x;
                int y;
            };
        """
        self.generic_visit(cursor)

    def visit_union_decl(self, cursor: clang.cindex.Cursor):
        """访问联合体声明
        示例：
            union Data {
                int i;
                float f;
                char str[20];
            };
        """
        self.generic_visit(cursor)

    def visit_enum_decl(self, cursor: clang.cindex.Cursor):
        """访问枚举声明
        示例：
            enum Color {
                RED,
                GREEN,
                BLUE
            };
        """
        self.generic_visit(cursor)

    def visit_field_decl(self, cursor: clang.cindex.Cursor):
        """访问字段声明，结构体、联合体或 C++ 类中的字段（在 C 中）或非静态数据成员（在 C++ 中）。"""
        self.generic_visit(cursor)

    def visit_enum_constant_decl(self, cursor: clang.cindex.Cursor):
        """访问枚举常量声明
        示例：
            enum Color {
                RED, // RED 是枚举常量
                GREEN,
                BLUE
            };
        """
        self.generic_visit(cursor)

    def visit_function_decl(self, cursor: clang.cindex.Cursor):
        """访问函数声明
        示例：
            int add(int a, int b) {
                return a + b;
            }
        """
        self.generic_visit(cursor)

    def visit_var_decl(self, cursor: clang.cindex.Cursor):
        """访问变量声明
        示例：
            int a = 10;
        """
        self.generic_visit(cursor)

    def visit_parm_decl(self, cursor: clang.cindex.Cursor):
        """访问函数参数声明"""
        self.generic_visit(cursor)

    def visit_typedef_decl(self, cursor: clang.cindex.Cursor):
        """访问类型定义声明
        示例：
            typedef int Integer;
        """
        self.generic_visit(cursor)

    def visit_alias_decl(self, cursor: clang.cindex.Cursor):
        """访问别名声明
        示例：
            using Integer = int;
        """
        self.generic_visit(cursor)

    def visit_type_ref(self, cursor: clang.cindex.Cursor):
        """访问类型引用"""
        self.generic_visit(cursor)

    def visit_member_ref(self, cursor: clang.cindex.Cursor):
        """访问成员引用，在某些非表达式上下文中出现的对结构体、联合体或类的成员的引用。
        """
        self.generic_visit(cursor)

    def visit_label_ref(self, cursor: clang.cindex.Cursor):
        """访问标签引用"""
        self.generic_visit(cursor)

    def visit_decl_ref_expr(self, cursor: clang.cindex.Cursor):
        """访问声明引用表达式"""
        self.generic_visit(cursor)

    def visit_variable_ref(self, cursor: clang.cindex.Cursor):
        """访问变量引用"""
        self.generic_visit(cursor)

    def visit_asm_stmt(self, cursor: clang.cindex.Cursor):
        """访问内联汇编语句"""
        self.generic_visit(cursor)

    def visit_member_ref_expr(self, cursor: clang.cindex.Cursor):
        """访问成员引用表达式"""
        self.generic_visit(cursor)

    def visit_call_expr(self, cursor: clang.cindex.Cursor):
        """访问函数调用表达式"""
        self.generic_visit(cursor)

    def visit_compound_stmt(self, cursor: clang.cindex.Cursor):
        """访问复合语句"""
        self.generic_visit(cursor)


class ClangCPPNodeVisitor(ClangNodeVisitor):

    def parse(self, path: str, args: List[str] = None, unsaved_files: List[Tuple[str, str]] = None, options=1,
              **kwargs):
        """解析代码生成语法树，默认支持 C++11 标准

        Args:
            path: str 代码文件路径
            args: List[str] 编译参数列表，默认值 ["-std=c++11"]
            unsaved_files: List[Tuple[str, str]] 未保存的文件内容，例如：[("a.c", "int main() { return 0; }")]
            options: int 解析选项，默认为 clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
                clang.cindex.TranslationUnit.PARSE_NONE: 无选项
                clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD: 使 AST 包含完整的预处理信息和宏定义
                clang.cindex.TranslationUnit.PARSE_INCOMPLETE: 允许不完整的代码（如仅有声明无定义的头文件）
                clang.cindex.TranslationUnit.PARSE_PRECOMPILED_PREAMBLE: 解析前将文件预编译并缓存，以加快后续解析速度
            kwargs: dict 其他参数


        Returns:
            clang.cindex.Cursor: 语法树根节点光标
        """
        args = args or ["-std=c++11"]
        return super().parse(path, args, unsaved_files, options, **kwargs)

    def visit_class_decl(self, cursor: clang.cindex.Cursor):
        """访问类声明
        示例：
            class Point {
            public:
                int x;
                int y;
            };
        """
        self.generic_visit(cursor)

    def visit_cxx_method(self, cursor: clang.cindex.Cursor):
        """访问 C++ 方法"""
        self.generic_visit(cursor)

    def visit_namespace(self, cursor: clang.cindex.Cursor):
        """访问命名空间"""
        self.generic_visit(cursor)

    def visit_linkage_spec(self, cursor: clang.cindex.Cursor):
        """访问链接规范
        例如：
            extern "C" {
                void foo();
            }
        """
        self.generic_visit(cursor)

    def visit_constructor(self, cursor: clang.cindex.Cursor):
        """访问构造函数"""
        self.generic_visit(cursor)

    def visit_destructor(self, cursor: clang.cindex.Cursor):
        """访问析构函数"""
        self.generic_visit(cursor)

    def visit_conversion_function(self, cursor: clang.cindex.Cursor):
        """访问转换函数
        例如：
            operator int() const;
        """
        self.generic_visit(cursor)

    def visit_template_type_parameter(self, cursor: clang.cindex.Cursor):
        """访问模板类型参数
        例如：
            template <typename T>
            void func(T t) {
                std::cout << t << std::endl;
            }
        """
        self.generic_visit(cursor)

    def visit_template_non_type_parameter(self, cursor: clang.cindex.Cursor):
        """访问模板非类型参数
        例如：
            template <int N>
            void func() {
                std::cout << N << std::endl;
            }
        """
        self.generic_visit(cursor)

    def visit_template_template_parameter(self, cursor: clang.cindex.Cursor):
        """访问模板模板参数
        例如：
            template <template <typename> class T>
            void func() {
                T<int> t;
            }
        """
        self.generic_visit(cursor)

    def visit_function_template(self, cursor: clang.cindex.Cursor):
        """访问函数模板
        例如：
            template <typename T>
            void func(T t) {
                std::cout << t << std::endl;
            }
        """
        self.generic_visit(cursor)

    def visit_class_template(self, cursor: clang.cindex.Cursor):
        """访问类模板
        例如：
            template <typename T>
            class A {
            public:
                T t;
            };
        """
        self.generic_visit(cursor)

    def visit_class_template_partial_specialization(self, cursor: clang.cindex.Cursor):
        """访问类模板部分特化
        例如：
            template <typename T>
            class A<T*> {
            public:
                T* t;
            };
        """
        self.generic_visit(cursor)

    def visit_namespace_alias(self, cursor: clang.cindex.Cursor):
        """访问命名空间别名
        例如：
            namespace A = B;
        """
        self.generic_visit(cursor)

    def visit_using_directive(self, cursor: clang.cindex.Cursor):
        """访问 using 指令
        例如：
            using namespace std;
        """
        self.generic_visit(cursor)

    def visit_using_declaration(self, cursor: clang.cindex.Cursor):
        """访问 using 声明
        例如：
            using std::cout;
        """
        self.generic_visit(cursor)

    def visit_cxx_access_specifier(self, cursor: clang.cindex.Cursor):
        """访问 C++ 访问说明符
        例如：
            class A {
            public:
                int a;
            protected:
                int b;
            private:
                int c;
            };
        """
        self.generic_visit(cursor)

    def visit_type_alias_template(self, cursor: clang.cindex.Cursor):
        """访问类型别名模板
        例如：
            template <typename T>
            using Vec = std::vector<T>;
        """
        self.generic_visit(cursor)

    def visit_template_ref(self, cursor: clang.cindex.Cursor):
        """访问模板引用"""
        self.generic_visit(cursor)

    def visit_namespace_ref(self, cursor: clang.cindex.Cursor):
        """访问命名空间引用"""
        self.generic_visit(cursor)

    def visit_overloaded_decl_ref(self, cursor: clang.cindex.Cursor):
        """访问重载声明引用"""
        self.generic_visit(cursor)

    def visit_cxx_base_specifier(self, cursor: clang.cindex.Cursor):
        """访问 C++ 基类说明符"""
        self.generic_visit(cursor)
