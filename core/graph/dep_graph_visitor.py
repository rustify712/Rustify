import os
from collections import defaultdict
from typing import Optional, List, Tuple

import clang

from core.graph.clang_visitor import ClangCPPNodeVisitor, ClangNodeVisitor, ClangCNodeVisitor
from core.graph.dep_graph import DGNode, DGEdge, DGEdgeType, DGNodeType, DepGraph


def get_definition_cursor(cursor: clang.cindex.Cursor) -> clang.cindex.Cursor:
    if cursor.kind == clang.cindex.CursorKind.TYPEDEF_DECL:
        # 区分 typedef xxx yyy; 和 typedef 匿名 yyy; 的情况
        related_declaration_cursor = cursor.underlying_typedef_type.get_declaration()
        if related_declaration_cursor.is_definition() and related_declaration_cursor.spelling == "":
            # typedef 匿名 yyy;
            return cursor
        elif related_declaration_cursor.is_definition():
            # 引用声明类型节点
            return related_declaration_cursor
        else:
            return cursor
    else:
        return cursor


def get_node_type_by_cursor(cursor: clang.cindex.Cursor) -> DGNodeType:
    """根据 Cursor 的类型获得 DGNodeType
    TRANSLATION_UNIT -> DGNodeType.FILE
    TYPEDEF_DECL -> DGNodeType.STRUCT/UNION/ENUM（匿名），TYPEDEF（类型定义）
    FUNCTION_DECL -> DGNodeType.FUNCTION
    CLASS_DECL -> DGNodeType.CLASS
    STRUCT_DECL -> DGNodeType.STRUCT
    UNION_DECL -> DGNodeType.UNION
    ENUM_DECL -> DGNodeType.ENUM
    """
    # cursor = get_definition_cursor(cursor)
    if cursor.kind == clang.cindex.CursorKind.TRANSLATION_UNIT:
        return DGNodeType.FILE
    elif cursor.kind == clang.cindex.CursorKind.TYPEDEF_DECL:
        # 匿名
        related_declaration_cursor = cursor.underlying_typedef_type.get_declaration()
        if related_declaration_cursor.is_definition() and related_declaration_cursor.spelling == "":
            return get_node_type_by_cursor(related_declaration_cursor)
        return DGNodeType.TYPEDEF
    elif cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL:
        return DGNodeType.FUNCTION
    elif cursor.kind == clang.cindex.CursorKind.CLASS_DECL:
        return DGNodeType.CLASS
    elif cursor.kind == clang.cindex.CursorKind.STRUCT_DECL:
        return DGNodeType.STRUCT
    elif cursor.kind == clang.cindex.CursorKind.UNION_DECL:
        return DGNodeType.UNION
    elif cursor.kind == clang.cindex.CursorKind.ENUM_DECL:
        return DGNodeType.ENUM
    elif cursor.kind == clang.cindex.CursorKind.TYPE_REF:
        return get_node_type_by_cursor(cursor.referenced)
    else:
        raise ValueError(f"不支持的节点类型: {cursor.kind}")


def is_reference_type(type_kind: clang.cindex.TypeKind) -> bool:
    """判断是否是引用类型"""
    return type_kind in [
        clang.cindex.TypeKind.POINTER,
        clang.cindex.TypeKind.LVALUEREFERENCE,
        clang.cindex.TypeKind.RVALUEREFERENCE,
        clang.cindex.TypeKind.MEMBERPOINTER
    ]


def is_primitive_type(type_kind: clang.cindex.TypeKind) -> bool:
    return type_kind in [
        clang.cindex.TypeKind.BOOL,
        clang.cindex.TypeKind.CHAR_U, clang.cindex.TypeKind.UCHAR, clang.cindex.TypeKind.CHAR16,
        clang.cindex.TypeKind.CHAR32,
        clang.cindex.TypeKind.USHORT, clang.cindex.TypeKind.UINT, clang.cindex.TypeKind.ULONG,
        clang.cindex.TypeKind.ULONGLONG, clang.cindex.TypeKind.UINT128,
        clang.cindex.TypeKind.CHAR_S, clang.cindex.TypeKind.SCHAR, clang.cindex.TypeKind.WCHAR,
        clang.cindex.TypeKind.SHORT, clang.cindex.TypeKind.INT, clang.cindex.TypeKind.LONG,
        clang.cindex.TypeKind.LONGLONG, clang.cindex.TypeKind.INT128,
        clang.cindex.TypeKind.FLOAT, clang.cindex.TypeKind.DOUBLE, clang.cindex.TypeKind.LONGDOUBLE,
        clang.cindex.TypeKind.FLOAT128,
        clang.cindex.TypeKind.NULLPTR
    ]


def is_complex_type(type_kind: clang.cindex.TypeKind) -> bool:
    return type_kind in [
        clang.cindex.TypeKind.RECORD,
        clang.cindex.TypeKind.ENUM,
        clang.cindex.TypeKind.TYPEDEF
    ]


class DepGraphClangNodeVisitor(ClangNodeVisitor):

    def __init__(self):
        super().__init__()
        self.translation_unit: Optional[clang.cindex.Cursor] = None
        self.nodes: List[DGNode] = []
        self.edges: List[DGEdge] = []

        self.parent_graph_node: Optional[DGNode] = None
        # 定义节点 cursor -> DGNode 节点
        self.cursor_node_map = {}

        # 声明节点ID -> 定义节点
        self.global_decl_def_map = {}
        # 未完成的边
        self.unfinished_edges = []

    @staticmethod
    def cursor_key(cursor: clang.cindex.Cursor):
        filename = cursor.location.file.name if cursor.location.file else ""
        line = cursor.location.line
        column = cursor.location.column
        kind = cursor.kind
        spelling = cursor.spelling
        return f"{filename}:{line}:{column}:{kind}:{spelling}"

    def set_cursor_map(self, cursor: clang.cindex.Cursor, node: DGNode):
        self.cursor_node_map[self.cursor_key(cursor)] = node

    def get_cursor_map(self, cursor: clang.cindex.Cursor):
        return self.cursor_node_map.get(self.cursor_key(cursor), None)

    def parent_node(self, cursor: clang.cindex.Cursor) -> Optional[DGNode]:
        """获取父节点"""
        if cursor.kind == clang.cindex.CursorKind.TRANSLATION_UNIT:
            return self.get_cursor_map(cursor)  # translation_unit
        if cursor.semantic_parent is None:
            parent_cursor = self.parent_cursor
        else:
            parent_cursor = cursor.semantic_parent

        if parent_cursor.kind == clang.cindex.CursorKind.TRANSLATION_UNIT:
            return self.get_cursor_map(parent_cursor)  # translation_unit

        while parent_cursor and parent_cursor.kind != clang.cindex.CursorKind.TRANSLATION_UNIT:
            if self.cursor_key(parent_cursor) in self.cursor_node_map:
                return self.get_cursor_map(parent_cursor)
            parent_cursor = parent_cursor.semantic_parent
        return self.get_cursor_map(self.translation_unit)  # translation_unit

    def add_node(self, node: DGNode):
        """添加节点"""
        if node in self.nodes:
            return
        self.nodes.append(node)

    def add_edge(self, src: DGNode, dst: DGNode, type: DGEdgeType):
        """添加边"""
        edge = DGEdge(src, dst, type)
        if edge in self.edges:
            return
        self.edges.append(edge)

    @staticmethod
    def from_language(language: str) -> "DepGraphClangNodeVisitor":
        """根据语言类型创建访问器

        Args:
            language: str 语言类型

        Returns:
            DepGraphClangNodeVisitor: 依赖图访问器
        """
        if language == "c":
            return DepGraphClangCNodeVisitor()
        elif language == "cpp":
            return DepGraphClangCPPNodeVisitor()
        else:
            raise ValueError("不支持的语言类型")

    @staticmethod
    def from_filepath(filepath: str, **kwargs) -> "DepGraphClangNodeVisitor":
        """根据文件路径创建访问器

        Args:
            filepath: str 文件路径

        Returns:
            DepGraphClangNodeVisitor: 依赖图访问器
        """
        allow_kwargs = [
            "args"
            "unsaved_files",
            "options",
            "expected_files"
        ]
        for key in kwargs:
            if key not in allow_kwargs:
                kwargs.pop(key)
        filename, postfix = os.path.splitext(filepath)
        visitor = DepGraphClangNodeVisitor.from_language(postfix.strip("."))
        root_node = visitor.parse(filepath, **kwargs)
        visitor.visit(root_node)
        return visitor

    def build_graph(self):
        """构建依赖图

        1. 将未完成的边添加到图中
        2. 处理 Import 边，将 .c 中的 Import 边转换为 .h 中的 Include 边
        """
        # 1. 将未完成的边添加到图中
        for src_id, dst_id, edge_type in self.unfinished_edges:
            src_node = DGNode.get_node_by_id(src_id)
            if src_node is None:
                src_def_node = self.global_decl_def_map.get(src_id, None)
                if src_def_node:
                    src_node = src_def_node
                else:
                    src_node_attr = DGNode._id2attr[src_id]
                    if src_node_attr[2] in self._expected_files:
                        print(f"src node not found: {DGNode._id2attr[src_id]} -> {DGNode._id2attr[dst_id]}")
                    continue
            dst_node = DGNode.get_node_by_id(dst_id)
            if dst_node is None:
                dst_def_node = self.global_decl_def_map.get(dst_id, None)
                if dst_def_node:
                    dst_node = dst_def_node
                else:
                    dst_node_attr = DGNode._id2attr[dst_id]
                    if dst_node_attr[2] in self._expected_files:
                        print(f"dst node not found: {DGNode._id2attr[src_id]} -> {DGNode._id2attr[dst_id]}")
                    # 将 ‘未找到 dst 节点’ 翻译为英文：
                    continue
            if src_node and dst_node:
                self.add_edge(src_node, dst_node, edge_type)
        # 2. 处理 Import 边，将 .c 中的 Import 边转换为 .h 中的 Include 边
        # header_node_map = {}
        # for node in self.nodes:
        #     if node.type == DGNodeType.FILE and node.name == node.location and node.name.endswith(".h"):
        #         # 这是一个 translation_unit 节点，即 C 头文件节点
        #         header_name = os.path.basename(node.name)
        #         header_node_map[header_name] = node
        # for node in self.nodes:
        #     if node.type == DGNodeType.FILE and node.name.endswith(".h") and node.location.endswith(".c"):
        #         print(node)

        dep_graph = DepGraph(self.nodes, self.edges)
        return dep_graph


class DepGraphClangCNodeVisitor(DepGraphClangNodeVisitor, ClangCNodeVisitor):

    def __init__(self):
        super().__init__()

    def visit_translation_unit(self, cursor: clang.cindex.Cursor):
        """访问语法树根节点"""
        graph_root_node = DGNode.new(
            type=DGNodeType.FILE,
            name=cursor.spelling,
            location=cursor.spelling,
            text=""
        )
        self.translation_unit = cursor
        self.set_cursor_map(cursor, graph_root_node)
        self.add_node(graph_root_node)

        self.parent_graph_node = graph_root_node
        self.parent_cursor = cursor
        self.generic_visit(self.node_children(cursor))
        self.parent_graph_node = None
        self.parent_cursor = None

    def visit_inclusion_directive(self, cursor: clang.cindex.Cursor):
        """访问包含指令，区分系统库头文件和用户自定义头文件
        仅记录用户自定义头文件
        """
        include_stmt_text = self.node_text(cursor).strip()
        if "<" in include_stmt_text and ">" in include_stmt_text:
            # 系统库头文件
            return
        include_node = DGNode.new(
            type=DGNodeType.FILE,
            name=cursor.spelling,
            location=cursor.location.file.name,
            text=include_stmt_text
        )
        self.set_cursor_map(cursor, include_node)
        self.add_node(include_node)
        self.add_edge(self.parent_node(self.translation_unit), include_node, DGEdgeType.IMPORT)

    def visit_macro_definition(self, cursor: clang.cindex.Cursor):
        """宏定义"""
        node_text = "#define " + self.node_text(cursor)
        macro_node = DGNode.new(
            type=DGNodeType.MACRO,
            name=cursor.spelling,
            location=cursor.location.file.name,
            text=node_text
        )
        self.set_cursor_map(cursor, macro_node)
        self.add_node(macro_node)
        self.add_edge(self.parent_node(self.translation_unit), macro_node, DGEdgeType.INCLUDE)

    def visit_var_decl(self, cursor: clang.cindex.Cursor):
        """访问变量声明
        仅记录全局变量
        """
        if cursor.semantic_parent.kind == clang.cindex.CursorKind.TRANSLATION_UNIT:
            var_node = DGNode.new(
                type=DGNodeType.VARIABLE,
                name=cursor.spelling,
                location=cursor.location.file.name,
                text=self.node_text(cursor)
            )
            self.set_cursor_map(cursor, var_node)
            self.add_node(var_node)
            self.add_edge(self.parent_node(cursor), var_node, DGEdgeType.INCLUDE)

        self.generic_visit(self.node_children(cursor))

    def visit_struct_decl(self, cursor: clang.cindex.Cursor):
        """结构体定义

        1. 匿名结构体定义
        typedef struct {} struct_name;
        2. 非定义节点
        struct undefined_struct_name;
        3. 有名结构体定义
        struct struct_name {};
        """
        # 1. 匿名结构体定义，交由 visit_typedef_decl 处理
        if cursor.is_anonymous() or cursor.spelling.strip() == "":
            return
        # 2. 声明节点
        if not cursor.is_definition():
            # 尝试查找定义节点
            def_cursor = cursor.get_definition()
            if def_cursor is None:
                return
            def_struct_node = DGNode.new(
                type=DGNodeType.STRUCT,
                name=def_cursor.spelling,
                location=def_cursor.location.file.name,
                text=self.node_text(def_cursor),
                extra={
                    "raw_comment": def_cursor.raw_comment or ""
                }
            )
            decl_struct_node_id = DGNode.generate_id(
                type=DGNodeType.DECLARATION,
                name=cursor.spelling,
                location=cursor.location.file.name,
                text=self.node_text(cursor),
                extra={
                    "type": DGNodeType.STRUCT,
                    "raw_comment": cursor.raw_comment or ""
                }
            )
            self.add_node(def_struct_node)
            self.add_edge(self.parent_node(def_cursor), def_struct_node, DGEdgeType.INCLUDE)
            self.global_decl_def_map[decl_struct_node_id] = def_struct_node
            return

        # 3. 有名结构体定义
        struct_node = DGNode.new(
            type=DGNodeType.STRUCT,
            name=cursor.spelling,
            location=cursor.location.file.name,
            text=self.node_text(cursor),
            extra={
                "raw_comment": cursor.raw_comment or ""
            }
        )
        self.set_cursor_map(cursor, struct_node)
        self.add_node(struct_node)
        self.add_edge(self.parent_node(cursor), struct_node, DGEdgeType.INCLUDE)

        parent_graph_node = self.parent_graph_node
        parent_cursor = self.parent_cursor
        self.parent_graph_node = struct_node
        self.parent_cursor = cursor
        self.generic_visit(self.node_children(cursor))
        self.parent_graph_node = parent_graph_node
        self.parent_cursor = parent_cursor

    def visit_union_decl(self, cursor: clang.cindex.Cursor):
        """访问联合体声明

        1. 匿名联合体定义
        typedef union {} union_name;
        2. 非定义节点
        union undefined_union_name;
        3. 有名联合体定义
        union union_name {};
        """
        # 1. 匿名联合体定义，交由 visit_typedef_decl 处理
        if cursor.is_anonymous() or cursor.spelling.strip() == "":
            return
        # 2. 非定义节点
        if not cursor.is_definition():
            # 尝试查找定义节点
            def_cursor = cursor.get_definition()
            if def_cursor is None:
                return
            def_union_node = DGNode.new(
                type=DGNodeType.UNION,
                name=def_cursor.spelling,
                location=def_cursor.location.file.name,
                text=self.node_text(def_cursor),
                extra={
                    "raw_comment": def_cursor.raw_comment or ""
                }
            )
            decl_union_node_id = DGNode.generate_id(
                type=DGNodeType.DECLARATION,
                name=cursor.spelling,
                location=cursor.location.file.name,
                text=self.node_text(cursor),
                extra={
                    "type": DGNodeType.UNION,
                    "raw_comment": cursor.raw_comment or ""
                }
            )
            self.add_node(def_union_node)
            self.add_edge(self.parent_node(def_cursor), def_union_node, DGEdgeType.INCLUDE)
            self.global_decl_def_map[decl_union_node_id] = def_union_node
            return
        # 3. 有名联合体定义
        union_node = DGNode.new(
            type=DGNodeType.UNION,
            name=cursor.spelling,
            location=cursor.location.file.name,
            text=self.node_text(cursor),
            extra={
                "raw_comment": cursor.raw_comment or ""
            }
        )
        self.set_cursor_map(cursor, union_node)
        self.add_node(union_node)
        self.add_edge(self.parent_node(cursor), union_node, DGEdgeType.INCLUDE)

        parent_graph_node = self.parent_graph_node
        parent_cursor = self.parent_cursor
        self.parent_graph_node = union_node
        self.parent_cursor = cursor
        self.generic_visit(self.node_children(cursor))
        self.parent_graph_node = parent_graph_node
        self.parent_cursor = parent_cursor

    def visit_enum_decl(self, cursor: clang.cindex.Cursor):
        """访问枚举声明

        1. 匿名枚举定义
        typedef enum {} enum_name;
        2. 非定义节点
        enum undefined_enum_name;
        3. 有名枚举定义
        enum enum_name {};
        """
        # 1. 匿名枚举定义，交由 visit_typedef_decl 处理
        if cursor.is_anonymous() or cursor.spelling.strip() == "":
            return
        # 2. 非定义节点
        if not cursor.is_definition():
            # 尝试查找定义节点
            def_cursor = cursor.get_definition()
            if def_cursor is None:
                return
            def_enum_node = DGNode.new(
                type=DGNodeType.ENUM,
                name=def_cursor.spelling,
                location=def_cursor.location.file.name,
                text=self.node_text(def_cursor),
                extra={
                    "raw_comment": def_cursor.raw_comment or ""
                }
            )
            decl_enum_node_id = DGNode.generate_id(
                type=DGNodeType.DECLARATION,
                name=cursor.spelling,
                location=cursor.location.file.name,
                text=self.node_text(cursor),
                extra={
                    "type": DGNodeType.ENUM,
                    "raw_comment": cursor.raw_comment or ""
                }
            )
            self.add_node(def_enum_node)
            self.add_edge(self.parent_node(def_cursor), def_enum_node, DGEdgeType.INCLUDE)
            self.global_decl_def_map[decl_enum_node_id] = def_enum_node
            return

        # 3. 有名枚举定义
        enum_node = DGNode.new(
            type=DGNodeType.ENUM,
            name=cursor.spelling,
            location=cursor.location.file.name,
            text=self.node_text(cursor),
            extra={
                "raw_comment": cursor.raw_comment or ""
            }
        )
        self.set_cursor_map(cursor, enum_node)
        self.add_node(enum_node)
        self.add_edge(self.parent_node(cursor), enum_node, DGEdgeType.INCLUDE)

        parent_graph_node = self.parent_graph_node
        parent_cursor = self.parent_cursor
        self.parent_graph_node = enum_node
        self.parent_cursor = cursor
        self.generic_visit(self.node_children(cursor))
        self.parent_graph_node = parent_graph_node
        self.parent_cursor = parent_cursor

    def visit_function_decl(self, cursor: clang.cindex.Cursor):
        """访问函数声明

        1. 函数声明
        int func_name(int a, int b);
        2. 函数定义
        int func_name(int a, int b) {}
        """
        # 1. 函数声明
        if not cursor.is_definition():
            # 尝试查找定义节点
            def_cursor = cursor.get_definition()
            if def_cursor is None:
                return
            def_function_node = DGNode.new(
                type=DGNodeType.FUNCTION,
                name=def_cursor.spelling,
                location=def_cursor.location.file.name,
                text=self.node_text(def_cursor),
                extra={
                    "raw_comment": def_cursor.raw_comment or "",
                    "param_names": [arg.spelling for arg in def_cursor.get_arguments()],
                    "param_types": [arg.type.spelling for arg in def_cursor.get_arguments()],
                    "return_type": def_cursor.result_type.spelling
                }
            )
            # 由于函数声明位置的不同，这里无需担心由于多态导致的重复添加
            decl_function_node_id = DGNode.generate_id(
                type=DGNodeType.DECLARATION,
                name=cursor.spelling,
                location=cursor.location.file.name,
                text=self.node_text(cursor),
                extra={
                    "type": DGNodeType.FUNCTION,
                    "raw_comment": cursor.raw_comment or ""
                }
            )
            self.set_cursor_map(def_cursor, def_function_node)
            self.add_node(def_function_node)
            self.add_edge(self.parent_node(def_cursor), def_function_node, DGEdgeType.INCLUDE)
            self.global_decl_def_map[decl_function_node_id] = def_function_node
            return
        # 2. 函数定义
        function_node = DGNode.new(
            type=DGNodeType.FUNCTION,
            name=cursor.spelling,
            location=cursor.location.file.name,
            text=self.node_text(cursor),
            extra={
                "raw_comment": cursor.raw_comment or "",
                "param_names": [arg.spelling for arg in cursor.get_arguments()],
                "param_types": [arg.type.spelling for arg in cursor.get_arguments()],
                "return_type": cursor.result_type.spelling
            }
        )
        self.set_cursor_map(cursor, function_node)
        self.add_node(function_node)
        self.add_edge(self.parent_node(cursor), function_node, DGEdgeType.INCLUDE)

        # 递归访问函数体
        parent_graph_node = self.parent_graph_node
        parent_cursor = self.parent_cursor
        self.parent_graph_node = function_node
        self.parent_cursor = cursor
        self.generic_visit(self.node_children(cursor))
        self.parent_graph_node = parent_graph_node
        self.parent_cursor = parent_cursor

    def visit_typedef_decl(self, cursor: clang.cindex.Cursor):
        """访问类型定义

        1. 匿名结构体/联合体/枚举定义
        typedef struct {} unnamed_struct;
        typedef union {} unnamed_union;
        typedef enum {} unnamed_enum;
        2. 有名类型定义
        2.1 原生类型
        typedef int int_type;
        typedef void *ArrayListValue;
        typedef int (*ArrayListEqualFunc)(ArrayListValue value1, ArrayListValue value2);
        2.2 复杂类型
        struct _struct_name {};
        typedef struct _struct_name struct_name;
        union _union_name {};
        typedef union _union_name union_name;
        typedef struct named_struct {} named_struct1;
        3.3 引用类型
        typedef struct _struct_name *struct_name_ptr;
        """
        underlying_type = cursor.underlying_typedef_type
        underlying_type_declaration = underlying_type.get_declaration()
        if underlying_type_declaration.is_definition() and underlying_type_declaration.spelling == "":
            # 1. 匿名结构体/联合体/枚举定义
            anonymous_node = DGNode.new(
                type=get_node_type_by_cursor(underlying_type_declaration),
                name=cursor.spelling,
                location=cursor.location.file.name,
                text=self.node_text(cursor),
                extra={
                    "raw_comment": cursor.raw_comment or ""
                }

            )
            self.set_cursor_map(cursor, anonymous_node)
            self.add_node(anonymous_node)
            self.add_edge(self.parent_node(cursor), anonymous_node, DGEdgeType.INCLUDE)

            parent_graph_node = self.parent_graph_node
            parent_cursor = self.parent_cursor
            self.parent_graph_node = anonymous_node
            self.parent_cursor = cursor
            self.generic_visit(self.node_children(cursor))
            self.parent_graph_node = parent_graph_node
            self.parent_cursor = parent_cursor
        else:
            # 对于struct、union、enum类型，需要通过 get_canonical 获取真实类型
            canonical_underlying_type = underlying_type.get_canonical()
            # 2. 类型定义
            typedef_node = DGNode.new(
                type=DGNodeType.TYPEDEF,
                name=cursor.spelling,
                location=cursor.location.file.name,
                text=self.node_text(cursor),
                extra={
                    "raw_comment": cursor.raw_comment or ""
                }
            )
            self.set_cursor_map(cursor, typedef_node)
            self.add_node(typedef_node)
            self.add_edge(self.parent_node(cursor), typedef_node, DGEdgeType.INCLUDE)
            if is_primitive_type(canonical_underlying_type.kind):
                # 2.1 原生类型，忽略
                ...
            elif is_complex_type(canonical_underlying_type.kind):
                # 2.2 复杂类型，忽略
                # 例如：typedef struct named_struct {} named_struct1;
                underlying_type_node_id = DGNode.generate_id(
                    type=get_node_type_by_cursor(underlying_type_declaration),
                    name=underlying_type_declaration.spelling,
                    location=underlying_type_declaration.location.file.name,
                    text=self.node_text(underlying_type_declaration)
                )
                underlying_type_node = DGNode.get_node_by_id(underlying_type_node_id)
                if underlying_type_node:
                    self.add_edge(typedef_node, underlying_type_node, DGEdgeType.INCLUDE)
            elif is_reference_type(canonical_underlying_type.kind):
                # 2.3 引用类型，忽略
                ...
            else:
                raise ValueError(f"不支持的类型定义: {canonical_underlying_type.kind}")

            parent_graph_node = self.parent_graph_node
            parent_cursor = self.parent_cursor
            self.parent_graph_node = typedef_node
            self.parent_cursor = cursor
            self.generic_visit(self.node_children(cursor))
            self.parent_graph_node = parent_graph_node
            self.parent_cursor = parent_cursor

    def visit_decl_ref_expr(self, cursor: clang.cindex.Cursor):
        """访问声明引用

        目前仅记录全局变量
        """
        def_cursor = cursor.get_definition() or cursor.referenced
        if def_cursor.kind == clang.cindex.CursorKind.VAR_DECL:
            if def_cursor.semantic_parent.kind == clang.cindex.CursorKind.TRANSLATION_UNIT:
                decl_node_id = DGNode.generate_id(
                    type=DGNodeType.VARIABLE,
                    name=def_cursor.spelling,
                    location=def_cursor.location.file.name,
                    text=self.node_text(def_cursor)
                )
                parent_node = self.parent_graph_node

                decl_node = DGNode.get_node_by_id(decl_node_id)
                if decl_node:
                    self.add_edge(parent_node, decl_node, DGEdgeType.INCLUDE)
                else:
                    self.unfinished_edges.append((
                        parent_node.id,
                        decl_node_id,
                        DGEdgeType.INCLUDE
                    ))

    def visit_type_ref(self, cursor: clang.cindex.Cursor):
        """访问类型引用

        1. 递归定义引用
        struct struct_name {
            struct struct_name *next;
        };
        2. 类型引用
        typedef struct struct_name struct_name;
        """

        def_cursor = cursor.get_definition() or cursor.referenced
        if def_cursor.is_definition():
            type_node_id = DGNode.generate_id(
                type=get_node_type_by_cursor(def_cursor),
                name=def_cursor.spelling,
                location=def_cursor.location.file.name,
                text=self.node_text(def_cursor)
            )
            parent_node = self.parent_node(cursor)
        else:
            type_node_id = DGNode.generate_id(
                type=DGNodeType.DECLARATION,
                name=cursor.type.get_declaration().spelling,
                location=cursor.type.get_declaration().location.file.name,
                text=self.node_text(cursor.type.get_declaration()),
                extra={
                    "type": get_node_type_by_cursor(cursor.type.get_declaration())
                }
            )
            parent_node = self.parent_node(cursor)

        if parent_node and parent_node.id == type_node_id:
            # 递归定义
            return
        type_node = DGNode.get_node_by_id(type_node_id)
        if type_node:
            self.add_edge(parent_node, type_node, DGEdgeType.INCLUDE)
        else:
            self.unfinished_edges.append((
                parent_node.id,
                type_node_id,
                DGEdgeType.INCLUDE
            ))

    def visit_member_ref_expr(self, cursor: clang.cindex.Cursor):
        """访问成员引用

        1. 结构体成员引用
        struct struct_name {
            int member;
        };
        void func() {
            struct_name obj;
            obj.member = 42;
        }
        """
        # 被引用成员的父节点
        referenced_parent_cursor = cursor.referenced.semantic_parent
        if referenced_parent_cursor.is_definition():
            referenced_parent_node_id = DGNode.generate_id(
                type=get_node_type_by_cursor(referenced_parent_cursor),
                name=referenced_parent_cursor.spelling,
                location=referenced_parent_cursor.location.file.name,
                text=self.node_text(referenced_parent_cursor)
            )
        else:
            referenced_parent_node_id = DGNode.generate_id(
                type=DGNodeType.DECLARATION,
                name=referenced_parent_cursor.spelling,
                location=referenced_parent_cursor.location.file.name,
                text=self.node_text(referenced_parent_cursor),
                extra={
                    "type": get_node_type_by_cursor(referenced_parent_cursor)
                }
            )
        referenced_parent_node = DGNode.get_node_by_id(referenced_parent_node_id)
        if referenced_parent_node:
            self.add_edge(self.parent_node(referenced_parent_cursor), referenced_parent_node, DGEdgeType.INCLUDE)
        else:
            self.unfinished_edges.append((
                self.parent_node(referenced_parent_cursor).id,
                referenced_parent_node_id,
                DGEdgeType.INCLUDE
            ))

    def visit_call_expr(self, cursor: clang.cindex.Cursor):
        """访问函数调用

        1. 函数调用
        int func_name(int a, int b) {
            printf("Hello, World!");
            return a + b;
        }
        void func() {
            int result = func_name(1, 2);
        }
        2. 递归调用
        void func1() {
            func1();
        }
        3. 函数指针调用
        void func2() {
            int (*func_ptr)(int, int) = func_name;
            int result = func_ptr(1, 2);
        }
        4. 函数参数调用
        typedef int (*PARAM_FUNC)(int, int);
        int func3(int a, PARAM_FUNC param_func) {
            return param_func(a, 2);
        }
        int func4(int a, int (*param_func1)(int, int) {
            return param_func1(a, 2);
        }
        """

        # 指针函数调用
        def_cursor = cursor.get_definition() or cursor.referenced
        if def_cursor and def_cursor.location.file.name not in self._expected_files:
            # 不在预期文件中，可能是系统库函数
            return
        if def_cursor is None:
            # 函数未定义
            return

        # 普通函数调用
        if def_cursor.kind == clang.cindex.CursorKind.FUNCTION_DECL:
            if def_cursor.is_definition():
                func_node_id = DGNode.generate_id(
                    type=DGNodeType.FUNCTION,
                    name=def_cursor.spelling,
                    location=def_cursor.location.file.name,
                    text=self.node_text(def_cursor)
                )
            else:
                func_node_id = DGNode.generate_id(
                    type=DGNodeType.DECLARATION,
                    name=def_cursor.spelling,
                    location=def_cursor.location.file.name,
                    text=self.node_text(def_cursor),
                    extra={
                        "type": DGNodeType.FUNCTION
                    }
                )
            parent_node = self.parent_graph_node
            if parent_node and parent_node.id == func_node_id:
                # 递归调用
                return
            func_node = DGNode.get_node_by_id(func_node_id)
            if func_node:
                self.add_edge(parent_node, func_node, DGEdgeType.INCLUDE)
            else:
                self.unfinished_edges.append((
                    parent_node.id,
                    func_node_id,
                    DGEdgeType.INCLUDE
                ))
        # 传参函数调用
        elif def_cursor.kind == clang.cindex.CursorKind.PARM_DECL:
            # 对于已经定义的 typedef 的函数指针
            if def_cursor.type == clang.cindex.TypeKind.TYPEDEF:
                def_cursor = def_cursor.type.get_declaration()
                if def_cursor.is_definition():
                    typedef_func_node_id = DGNode.generate_id(
                        type=DGNodeType.TYPEDEF,
                        name=def_cursor.spelling,
                        location=def_cursor.location.file.name,
                        text=self.node_text(def_cursor)
                    )
                else:
                    typedef_func_node_id = DGNode.generate_id(
                        type=DGNodeType.DECLARATION,
                        name=def_cursor.spelling,
                        location=def_cursor.location.file.name,
                        text=self.node_text(def_cursor),
                        extra={
                            "type": DGNodeType.TYPEDEF
                        }
                    )
                typedef_func_node = DGNode.get_node_by_id(typedef_func_node_id)
                if typedef_func_node:
                    self.add_edge(self.parent_graph_node, typedef_func_node, DGEdgeType.INCLUDE)
                else:
                    self.unfinished_edges.append((
                        self.parent_graph_node.id,
                        typedef_func_node_id,
                        DGEdgeType.INCLUDE
                    ))
        elif def_cursor.kind == clang.cindex.CursorKind.VAR_DECL:
            # TODO: 函数指针调用
            # 获得指针指向的类型
            if def_cursor.type.kind == clang.cindex.TypeKind.POINTER:
                pointee_type = def_cursor.type.get_pointee()
                if pointee_type.kind == clang.cindex.TypeKind.FUNCTIONPROTO:
                    # 查找初始化表达式
                    func_cursor = None
                    for child in self.node_children(def_cursor):
                        if child.kind == clang.cindex.CursorKind.UNEXPOSED_EXPR:
                            for grandchild in self.node_children(child):
                                if grandchild.kind == clang.cindex.CursorKind.DECL_REF_EXPR:
                                    func_cursor = grandchild.get_definition()
                                    break
                        elif child.kind == clang.cindex.CursorKind.DECL_REF_EXPR:
                            func_cursor = child
                            break
                    if func_cursor:
                        if func_cursor.is_definition():
                            func_node_id = DGNode.generate_id(
                                type=DGNodeType.FUNCTION,
                                name=func_cursor.spelling,
                                location=func_cursor.location.file.name,
                                text=self.node_text(func_cursor)
                            )
                        else:
                            func_node_id = DGNode.generate_id(
                                type=DGNodeType.DECLARATION,
                                name=func_cursor.spelling,
                                location=func_cursor.location.file.name,
                                text=self.node_text(func_cursor),
                                extra={
                                    "type": DGNodeType.FUNCTION
                                }
                            )
                        func_node = DGNode.get_node_by_id(func_node_id)
                        parent_node = self.parent_graph_node
                        if func_node:
                            self.add_edge(parent_node, func_node, DGEdgeType.INCLUDE)
                        else:
                            self.unfinished_edges.append((
                                parent_node.id,
                                func_node_id,
                                DGEdgeType.INCLUDE
                            ))

        # 不改变 self.parent_graph_node
        parent_cursor = self.parent_cursor
        self.parent_cursor = cursor
        self.generic_visit(self.node_children(cursor))
        self.parent_cursor = parent_cursor


class DepGraphClangCPPNodeVisitor(DepGraphClangCNodeVisitor, ClangCPPNodeVisitor):

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
        args = args or ["-std=c++11"]
        return super().parse(path, args, unsaved_files, options, **kwargs)

    def visit_class_decl(self, cursor: clang.cindex.Cursor):
        """访问类声明"""
        if not cursor.is_definition():
            # 尝试查找定义节点
            def_cursor = cursor.get_definition()
            if def_cursor is None:
                return
            def_class_node = DGNode.new(
                type=DGNodeType.CLASS,
                name=def_cursor.spelling,
                location=def_cursor.location.file.name,
                text=self.node_text(def_cursor),
                extra={
                    "raw_comment": def_cursor.raw_comment or ""
                }
            )
            decl_class_node_id = DGNode.generate_id(
                type=DGNodeType.DECLARATION,
                name=cursor.spelling,
                location=cursor.location.file.name,
                text=self.node_text(cursor),
                extra={
                    "type": DGNodeType.CLASS,
                    "raw_comment": cursor.raw_comment or ""
                }
            )
            self.set_cursor_map(def_cursor, def_class_node)
            self.add_node(def_class_node)
            self.add_edge(self.parent_node(def_cursor), def_class_node, DGEdgeType.INCLUDE)
            self.global_decl_def_map[decl_class_node_id] = def_class_node
            return

        class_node = DGNode.new(
            type=DGNodeType.CLASS,
            name=cursor.spelling,
            location=cursor.location.file.name,
            text=self.node_text(cursor),
            extra={
                "raw_comment": cursor.raw_comment or ""
            }
        )
        self.set_cursor_map(cursor, class_node)
        self.add_node(class_node)
        self.add_edge(self.parent_node(cursor), class_node, DGEdgeType.INCLUDE)

        parent_graph_node = self.parent_graph_node
        parent_cursor = self.parent_cursor
        self.parent_graph_node = class_node
        self.parent_cursor = cursor
        self.generic_visit(self.node_children(cursor))
        self.parent_graph_node = parent_graph_node
        self.parent_cursor = parent_cursor

    def visit_constructor(self, cursor: clang.cindex.Cursor):
        """访问构造函数"""
        if not cursor.is_definition():
            # 非定义节点，忽略
            return
        constructor_node = DGNode.new(
            type=DGNodeType.CONSTRUCTOR,
            name=cursor.spelling,
            location=cursor.location.file.name,
            text=self.node_text(cursor),
            extra={
                "raw_comment": cursor.raw_comment or "",
                "param_names": [arg.spelling for arg in cursor.get_arguments()],
                "param_types": [arg.type.spelling for arg in cursor.get_arguments()]
            }
        )
        self.set_cursor_map(cursor, constructor_node)
        self.add_node(constructor_node)
        self.add_edge(self.parent_node(cursor), constructor_node, DGEdgeType.INCLUDE)

        parent_graph_node = self.parent_graph_node
        parent_cursor = self.parent_cursor
        self.parent_graph_node = constructor_node
        self.parent_cursor = cursor
        self.generic_visit(self.node_children(cursor))
        self.parent_graph_node = parent_graph_node
        self.parent_cursor = parent_cursor

    def visit_destructor(self, cursor: clang.cindex.Cursor):
        """访问析构函数"""
        if not cursor.is_definition():
            # 非定义节点，忽略
            return
        destructor_node = DGNode.new(
            type=DGNodeType.DESTRUCTOR,
            name=cursor.spelling,
            location=cursor.location.file.name,
            text=self.node_text(cursor),
            extra={
                "raw_comment": cursor.raw_comment or ""
            }
        )
        self.set_cursor_map(cursor, destructor_node)
        self.add_node(destructor_node)
        self.add_edge(self.parent_node(cursor), destructor_node, DGEdgeType.INCLUDE)

        parent_graph_node = self.parent_graph_node
        parent_cursor = self.parent_cursor
        self.parent_graph_node = destructor_node
        self.parent_cursor = cursor
        self.generic_visit(self.node_children(cursor))
        self.parent_graph_node = parent_graph_node
        self.parent_cursor = parent_cursor
