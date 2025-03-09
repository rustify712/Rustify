import copy
import uuid
from collections import defaultdict, deque
from enum import Enum
from typing import Callable, List, Optional, Dict, Tuple, Generator

from graphviz import Digraph

DGNODE_NAMESPACE = uuid.uuid4()


class DGEdgeType(Enum):
    """依赖图边类型

    IMPORT: 文件节点导入关系
    INCLUDE: 文件节点/函数节点/类节点包含关系
    ALIAS: 类型别名关系
    """
    IMPORT = "import"
    INCLUDE = "include"


class DGNodeType(Enum):
    """依赖图节点类型

    FILE: 文件
    DECLARATION: 声明
    Type: 类型
    TYPEDEF: 类型别名
    FUNCTION: 函数
    CONSTRUCTOR: 构造函数
    DESTRUCTOR: 析构函数
    CALL_EXPRESSION: 函数调用
    CLASS: 类
    STRUCT: 结构体
    ENUM: 枚举
    VARIABLE: 变量
    MACRO: 宏
    MACRO_FUNCTION: 宏函数
    MACRO_IFDEF: 宏条件编译

    TEMP: 临时节点
    """
    FILE = "file"
    DECLARATION = "declaration"
    TYPE = "type"
    TYPEDEF = "typedef"
    FUNCTION = "function"
    CONSTRUCTOR = "constructor"
    DESTRUCTOR = "destructor"
    CLASS = "class"
    STRUCT = "struct"
    UNION = "union"
    ENUM = "enum"
    VARIABLE = "variable"
    MACRO = "macro"
    MACRO_FUNCTION = "macro_function"
    MACRO_IFDEF = "macro_ifdef"


class DGNode:
    """依赖图节点

    Attributes:
        type: str 节点类型
        name: str 节点名称
        location: str 节点所属文件
        text: str 节点文本
        extra: dict 额外信息
        edges: list[DGEdge] 出边列表
        src_edges: list[DGEdge] 入边列表

    Warnings:
        1. 为了避免重复创建节点，需要使用 new 方法创建节点实例
    """

    _instances = {}
    _id2attr = {}

    def __init__(self, id: str, type: DGNodeType, name: str, location: str, text: str,
                 extra: Optional[dict] = None):
        if id in self._instances:
            raise ValueError(f"节点 {id} 已存在, 请使用 new 方法创建节点实例")
        else:
            self._instances[id] = self
        self.id = id
        self.type = type
        self.name = name
        self.text = text
        self.location = location
        self.extra = extra or {}
        self.edges = []  # 出边
        self.src_edges = []  # 入边

    @classmethod
    def generate_id(cls, type: DGNodeType, name: str, location: str, text: str, extra: Optional[dict] = None) -> str:
        """生成节点 ID

        Args:
            extra:
            type: DGNodeType 节点类型
            name: str 节点名称
            location: str 节点所属文件
            text: str 节点文本

        Returns:
            str: 节点 ID
        """
        if extra is None:
            extra = dict()
        if type == DGNodeType.DECLARATION:
            # 声明节点
            assert "type" in extra, "声明节点 extra 中必须包含 type 字段"
            text = extra["type"]
        id = str(uuid.uuid5(DGNODE_NAMESPACE, f"{type}@{name}@{location}@{text}"))
        cls._id2attr[id] = (type, name, location, text)
        return id

    @classmethod
    def new(cls, type: DGNodeType, name: str, location: str, text: str,
            extra: Optional[dict] = None):
        """创建节点实例，若节点已存在则返回已存在的节点

        Args:
            type: DGNodeType 节点类型
            name: str 节点名称
            location: str 节点所属文件
            text: str 节点文本
            extra: Optional[dict] 额外信息

        Returns:
            DGNode: 节点
        """
        id = cls.generate_id(type, name, location, text, extra)
        if id in cls._instances:
            return cls._instances[id]
        instance = DGNode(id, type, name, location, text, extra)
        cls._instances[id] = instance
        return instance

    @classmethod
    def get_node_by_id(cls, id: str) -> Optional["DGNode"]:
        return cls._instances.get(id, None)

    def add_edge(self, dst: "DGNode", type: DGEdgeType):
        """建立关系

        Args:
            dst: DGNode 目标节点
            type: DGEdgeType 边类型
        """
        edge = DGEdge(self, dst, type)
        if edge in self.edges:
            return None
        self.edges.append(edge)
        dst.src_edges.append(edge)
        return edge

    def search_edge_by(self, func: Callable[["DGEdge"], bool]) -> Optional["DGEdge"]:
        """根据条件查找边, 返回第一个符合条件的出边

        Args:
            func: Callable[[DGEdge], bool] 条件函数

        Returns:
            Optional[DGEdge]: 边
        """
        for edge in self.edges:
            if not func(edge):
                continue
            return edge
        return None

    def search_src_edge_by(self, func: Callable[["DGEdge"], bool]) -> Optional["DGEdge"]:
        """根据条件查找边, 返回第一个符合条件的入边

        Args:
            func: Callable[[DGEdge], bool] 条件函数

        Returns:
            Optional[DGEdge]: 边
        """
        for edge in self.src_edges:
            if not func(edge):
                continue
            return edge
        return None

    def find_edges_by(self, func: Callable[["DGEdge"], bool]) -> List["DGEdge"]:
        """根据条件查找边, 返回所有符合条件的出边

        Args:
            func: Callable[[DGEdge], bool] 条件函数

        Returns:
            List[DGEdge]: 边列表
        """
        related_edges = []
        for edge in self.edges:
            if not func(edge):
                continue
            related_edges.append(edge)
        return related_edges

    def find_src_edges_by(self, *, func: Callable[["DGEdge"], bool]) -> List["DGEdge"]:
        """根据条件查找边, 返回所有符合条件的入边

        Args:
            func: Callable[[DGEdge], bool] 条件函数

        Returns:
            List[DGEdge]: 边列表
        """
        related_edges = []
        for edge in self.src_edges:
            if not func(edge):
                continue
            related_edges.append(edge)
        return related_edges

    def search_node_by(
            self,
            func: Callable[["DGNode"], bool]
    ) -> Optional["DGNode"]:
        """根据条件查找节点, 返回第一个符合条件的节点

        Args:
            func: Callable[[DGNode], bool] 条件函数

        Returns:
            Optional[DGNode]: 节点
        """
        for edge in self.edges:
            if not func(edge.dst):
                continue
            return edge.dst
        return None

    def find_nodes_by(
            self,
            func: Callable[["DGNode"], bool]
    ) -> List["DGNode"]:
        """根据条件查找节点, 返回所有符合条件的节点

        Args:
            func: Callable[[DGNode], bool] 条件函数

        Returns:
            List[DGNode]: 节点列表
        """
        related_nodes = []
        for edge in self.edges:
            if not func(edge.dst):
                continue
            related_nodes.append(edge.dst)
        return related_nodes

    def has_node(
            self,
            func: Callable[["DGNode"], bool]
    ):
        """判断是否存在节点

        Args:
            func: Callable[[DGNode], bool] 条件函数

        Returns:
            bool: 是否存在
        """
        return bool(self.search_node_by(func))

    @property
    def is_leaf(self):
        return len(self.edges) == 0

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    def __repr__(self):
        return f"DGNode(name={self.name}, type={self.type}, location={self.location})"


class DGEdge:
    """依赖图边

    Attributes:
        src: DGNode 源节点
        dst: DGNode 目标节点
        type: str 边类型
    """

    def __init__(self, src: DGNode, dst: DGNode, type: DGEdgeType):
        self.src = src
        self.dst = dst
        self.type = type

    def remove_edge(self):
        self.src.edges.remove(self)
        self.dst.src_edges.remove(self)

    def __eq__(self, other):
        return self.src == other.src and self.dst == other.dst and self.type == other.type

    def __repr__(self):
        return f"DGEdge({self.src.name} -> {self.dst.name}, {self.type})"


class DepGraph:
    """依赖图

    Attributes:
        nodes: List[DGNode] 节点列表
        edges: List[DGNode] 边列表
    """

    def __init__(self, nodes: List[DGNode], edges: List[DGEdge]):
        nodes, edges = self.build(nodes, edges)
        self.nodes = nodes
        self.edges = edges
        self.node_id_map = {
            node.id: node
            for node in nodes
        }

        for edge in edges:
            self.node_id_map[edge.src.id].edges.append(edge)
            self.node_id_map[edge.dst.id].src_edges.append(edge)
            edge.src = self.node_id_map[edge.src.id]
            edge.dst = self.node_id_map[edge.dst.id]

    def add_node(self, node: DGNode):
        """添加节点

        Args:
            node: DGNode 节点
        """
        if node not in self.nodes:
            self.nodes.append(node)

    def remove_node(self, node: DGNode) -> bool:
        """删除节点，并同时删除相关的边

        Args:
            node: DGNode 节点

        Returns:
            bool: 是否删除成功
        """
        if node not in self.nodes:
            return False
        self.nodes.remove(node)
        for edge in node.edges:
            edge.dst.src_edges.remove(edge)
        for edge in node.src_edges:
            edge.src.edges.remove(edge)
        return True

    def add_edge(self, src: DGNode, dst: DGNode, type: DGEdgeType):
        """添加节点

        Args:
            src: DGNode 源节点
            dst: DGNode 目标节点
            type: DGEdgeType 边类型
        """
        self.add_node(src)
        self.add_node(dst)
        edge = src.add_edge(dst, type)
        if edge:
            self.edges.append(edge)

    def remove_edge(self, edge: DGEdge):
        """删除依赖关系

        Args:
            edge: DGEdge 边
        """
        edge.remove_edge()
        if edge in self.edges:
            self.edges.remove(edge)

    def search_node_by(self, func: Callable[[DGNode], bool]) -> Optional[DGNode]:
        """根据条件查找节点, 返回第一个符合条件的节点

        Args:
            func: Callable[[DGNode], bool] 条件函数

        Returns:
            Optional[DGNode]: 节点
        """
        for node in self.nodes:
            if not func(node):
                continue
            return node
        return None

    def find_nodes_by(self, func: Callable[[DGNode], bool]) -> List[DGNode]:
        """根据条件查找节点, 返回所有符合条件的节点

        Args:
            func: Callable[[DGNode], bool] 条件函数

        Returns:
            List[DGNode]: 节点列表
        """
        related_nodes = []
        for node in self.nodes:
            if not func(node):
                continue
            related_nodes.append(node)
        return related_nodes

    def build(self, nodes: Optional[List[DGNode]], edges: Optional[List[DGEdge]]):
        """构建依赖图

        1. 检查是否存在环，若存在，则解环：删除最后的依赖关系
        2. 传递性缩减算法
        Returns:
            DepGraph: 依赖图
        """
        # 1. 检查是否存在环，若存在，则解环：删除最后的依赖关系
        nodes, edges = self.check_and_break_cycle(nodes, edges)
        # 2. 传递性缩减算法
        adj_list = defaultdict(lambda: defaultdict(list))
        edge_types = set(edge.type for edge in edges)
        in_degree = defaultdict(int)
        for edge in edges:
            adj_list[edge.src][edge.type].append(edge.dst)
            in_degree[edge.dst] += 1
        # 2.1 拓扑排序
        topo_order = []
        zero_in_degree = deque([node for node in nodes if in_degree[node] == 0])
        while zero_in_degree:
            node = zero_in_degree.pop()
            topo_order.append(node)
            for edge_type in edge_types:
                for neighbor in adj_list[node][edge_type]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        zero_in_degree.append(neighbor)

        # 2.2 传递性缩减
        reduced_edges = []
        reachable = {edge_type: {node: set() for node in nodes} for edge_type in edge_types}
        for node in reversed(topo_order):
            for edge_type in adj_list[node]:
                for neighbor in adj_list[node][edge_type]:
                    reachable[edge_type][node].update(reachable[edge_type][neighbor])
                for neighbor in adj_list[node][edge_type]:
                    if neighbor not in reachable[edge_type][node]:
                        reduced_edges.append(DGEdge(node, neighbor, edge_type))
                        reachable[edge_type][node].add(neighbor)
                    # reachable[edge_type][node].update(reachable[edge_type][neighbor])

        return nodes, reduced_edges

    @classmethod
    def check_and_break_cycle(cls, nodes: List[DGNode], edges: List[DGEdge]) -> Tuple[List[DGNode], List[DGEdge]]:
        """检查是否存在环，若存在，则解环：删除最后的依赖关系

        Args:
            nodes: List[DGNode] 节点列表
            edges: List[DGEdge] 边列表

        Returns:
            List[DGNode]: 无环节点
        """
        nodes = copy.deepcopy(nodes)
        edges = copy.deepcopy(edges)

        def find_cycles():
            """使用深度优先搜索检测所有环"""
            visited = set()
            stack = []  # DFS 栈
            in_stack = set()  # 栈中节点集合，用于判断回边
            cycles = []  # 检测到的所有环

            adj_list = defaultdict(list)
            for edge in edges:
                adj_list[edge.src].append(edge.dst)

            def dfs(node):
                if node in visited:
                    return
                stack.append(node)
                in_stack.add(node)
                visited.add(node)

                for neighbor in adj_list[node]:
                    if neighbor not in visited:
                        dfs(neighbor)
                    elif neighbor in in_stack:
                        # 检测到环
                        cycle_start_index = stack.index(neighbor)
                        cycle = stack[cycle_start_index:] + [neighbor]
                        cycle_edges = [
                            edge for edge in edges if edge.src in cycle and edge.dst in cycle
                        ]
                        cycles.append(cycle_edges)

                stack.pop()
                in_stack.remove(node)

            for node in nodes:
                if node not in visited:
                    dfs(node)

            return cycles

        while True:
            all_cycles = find_cycles()

            if not all_cycles:
                break

            for cycle_edges in all_cycles:
                if not cycle_edges:
                    continue
                edges.remove(cycle_edges[-1])
                break

        return nodes, edges

    def traverse_leaf_nodes_in_module(self, nodes: List[DGNode]):
        """拓扑排序模块内的节点， 并适当合并部分节点"""
        visited_nodes = set()

        def merge_linked_nodes(node: DGNode):
            """合并双向单依赖节点
            仅考虑以下情况：
                1. 单依赖边的两端顶点为 DGNodeType.FUNCTION
                2. 单依赖边的 src 顶点为 DGNodeType.TYPEDEF, dst 顶点为 DGNodeType.STRUCT, DGNodeType.UNION, DGNodeType.ENUM
            """

            if len(node.src_edges) != 1:
                return [node]
            src_edge = node.src_edges[0]
            # 1. 单依赖边的两端顶点为 DGNodeType.FUNCTION
            if not ((src_edge.src.type == DGNodeType.FUNCTION and src_edge.dst.type == DGNodeType.FUNCTION) or
                    # 2. 单依赖边的 src 顶点为 DGNodeType.TYPEDEF, dst 顶点为 DGNodeType.STRUCT, DGNodeType.UNION, DGNodeType.ENUM
                    (src_edge.src.type == DGNodeType.TYPEDEF and src_edge.dst.type in [DGNodeType.STRUCT,
                                                                                       DGNodeType.UNION,
                                                                                       DGNodeType.ENUM])
            ):
                return [node]
            src_node = node.src_edges[0].src
            if len([
                edge
                for edge in src_node.edges
                if edge.dst not in visited_nodes
            ]) != 1:
                return [node]
            # 双向单依赖
            return [node] + merge_linked_nodes(src_node)

        while len(nodes) - len(visited_nodes) != 0:
            leaf_nodes = [
                node for node in nodes
                if node not in visited_nodes and all([edge.dst in visited_nodes for edge in node.edges])
            ]
            yield_nodes = []
            if not leaf_nodes:
                raise Exception("存在环路, 剩余节点：", set(nodes) - visited_nodes)
            # 向上寻找叶子节点的父节点，若单依赖，则合并
            for leaf_node in leaf_nodes:
                yield_nodes.append(merge_linked_nodes(leaf_node))

            for yield_node in yield_nodes:
                visited_nodes.update(yield_node)

            yield yield_nodes

    def traverse_modules(self) -> Generator[Tuple[str, List[List[List[DGNode]]]], None, None]:
        """依次遍历所有出度为 0 的节点, 忽略临时节点, 按模块划分

        Returns:
            Generator[Tuple[str, List[List[List[DGNode]]]], None, None]: 模块名，模块内节点列表
        """
        # 构建文件间的依赖图
        file_dep_graph = defaultdict(set)
        for edge in self.edges:
            src_file = edge.src.location
            dst_file = edge.dst.location
            file_dep_graph[src_file].add(dst_file)

        file_groups = []
        for src_file, dst_files in file_dep_graph.items():
            is_merged = False
            for file_group in file_groups:
                if src_file in file_group or dst_files & file_group:
                    is_merged = True
                    file_group.update(dst_files | {src_file})
            if not is_merged:
                file_groups.append(dst_files | {src_file})

        # 根据 file_groups 对 DGNode 进行划分
        file_group_id_map = {}
        groups = defaultdict(list)
        for index, file_group in enumerate(file_groups):
            for file in file_group:
                file_group_id_map[file] = index

        for node in self.nodes:
            if node.type == DGNodeType.FILE:
                continue
            group_id = file_group_id_map[node.location]
            groups[group_id].append(node)

        # 对模块内的节点进行拓扑排序，并适当合并节点
        for group_id, nodes in groups.items():
            yield {
                "module": group_id,
                "files": file_groups[group_id],
            }, self.traverse_leaf_nodes_in_module(nodes)

    def print(self):
        """按模块打印依赖图

        """
        visited_nodes = []
        nodes = self.nodes

        def _print_node(_node: DGNode, depth: int):
            if _node not in visited_nodes:
                print(f"\033[1;32m{'  ' * depth}Node: {_node}\033[0m")
                visited_nodes.append(_node)
            for edge in _node.edges:
                print(
                    f"\033[1;32m{'  ' * (depth + 1)}{edge.type} -> {edge.dst}\033[0m")
                if edge.dst not in visited_nodes:
                    print(f"{edge.dst.text}")
                    # 避免打印 Node: xxx
                    visited_nodes.append(edge.dst)
                _print_node(edge.dst, depth + 1)

        _print_node(nodes[0], 0)
        return visited_nodes

    def render(self, filename: str):
        """渲染依赖图

        Args:
            filename: str 文件名
        """
        # 递归绘制依赖图，会导致未被依赖的节点丢失
        visited_nodes = []

        dot = Digraph()
        for node in self.nodes:
            node_attrs = {}
            if node.type == DGNodeType.FILE:
                node_attrs["shape"] = "folder"
            elif node.type == DGNodeType.TYPEDEF:
                node_attrs["shape"] = "note"
            dot.node(str(node.id), label=node.name, _attributes=node_attrs)
        for edge in self.edges:
            edge_attrs = {}
            dot.edge(str(edge.src.id), str(edge.dst.id), label=edge.type.value, _attributes=edge_attrs)

        dot.render(filename, format="png", cleanup=True)
        return visited_nodes
