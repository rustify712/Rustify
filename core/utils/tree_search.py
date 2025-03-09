import json
import math
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional


class TreeSearchNode:

    def __init__(self, data: dict, parent: Optional["TreeSearchNode"] = None):
        self.id = str(uuid.uuid4())
        self.data = data
        self.parent = parent
        self.value = None
        self.children: list["TreeSearchNode"] = []

        # MCTS
        self.visit_count = 0
        self.mcts_value = 0.0


class SearchStrategy(ABC):
    """搜索策略基类

    Args:
        expansion_func (Callable[[TreeSearchNode], list[dict]]): 扩展节点的函数
        simulation_func (Callable[[TreeSearchNode], float]): 模拟节点的函数
        compare_func (Callable[[TreeSearchNode, TreeSearchNode], bool]): 比较节点的函数, True 表示后一个节点更优
        max_depth (int, optional): 最大搜索深度. Defaults to 5.
        max_expansion (int, optional): 最大扩展次数. Defaults to -1.
    """
    strategy_name: str

    def __init__(
            self,
            expansion_func: Callable[[TreeSearchNode], list[dict]],
            simulation_func: Callable[[TreeSearchNode], float],
            compare_func: Callable[[TreeSearchNode, TreeSearchNode], bool],
            max_depth: int = 5,
            max_expansion: int = -1
    ):
        self.expansion_func = expansion_func
        self.simulation_func = simulation_func
        self.compare_func = compare_func
        self.max_depth = max_depth
        self.max_expansion = max_expansion

        self.expansion_count = 0

    @abstractmethod
    def search(self, root: TreeSearchNode, target: Callable[[TreeSearchNode], bool]) -> TreeSearchNode:
        raise NotImplementedError

    @abstractmethod
    def expand(self, node: TreeSearchNode):
        raise NotImplementedError


class DFSSearchStrategy(SearchStrategy):
    strategy_name = "DFS"

    def search(self, root: TreeSearchNode, target: Callable[[TreeSearchNode], bool]) -> TreeSearchNode:
        stack = [(root, 0)]
        best_node = root
        while stack:
            node, depth = stack.pop()
            node.value = self.simulation_func(node)
            if target(node):
                return node
            if self.compare_func(best_node, node):
                best_node = node
            if depth < self.max_depth:
                self.expand(node)
                for child in reversed(node.children):
                    stack.append((child, depth + 1))
            if 0 < self.max_expansion <= self.expansion_count:
                break
        return best_node

    def expand(self, node: TreeSearchNode):
        for child_data in self.expansion_func(node):
            child = TreeSearchNode(child_data, parent=node)
            node.children.append(child)
            self.expansion_count += 1
            if 0 < self.max_expansion <= self.expansion_count:
                break


class BFSSearchStrategy(SearchStrategy):
    strategy_name = "BFS"

    def search(self, root: TreeSearchNode, target: Callable[[TreeSearchNode], bool]) -> TreeSearchNode:
        queue = [(root, 0)]
        best_node = root
        while queue:
            node, depth = queue.pop(0)
            node.value = self.simulation_func(node)
            if target(node):
                return node
            if self.compare_func(best_node, node):
                best_node = node
            if depth < self.max_depth:
                self.expand(node)
                for child in node.children:
                    queue.append((child, depth + 1))
            if 0 < self.max_expansion <= self.expansion_count:
                break
        return best_node

    def expand(self, node: TreeSearchNode):
        for child_data in self.expansion_func(node):
            child = TreeSearchNode(child_data, parent=node)
            node.children.append(child)
            self.expansion_count += 1
            if 0 < self.max_expansion <= self.expansion_count:
                break


class GreedySearchStrategy(SearchStrategy):
    strategy_name = "Greedy"

    def __init__(
            self,
            expansion_func: Callable[[TreeSearchNode], list[dict]],
            simulation_func: Callable[[TreeSearchNode], float],
            compare_func: Callable[[TreeSearchNode, TreeSearchNode], bool],
            selection_func: Callable[[list[TreeSearchNode]], TreeSearchNode],
            max_depth: int = 5,
            max_expansion: int = -1
    ):
        super().__init__(expansion_func, simulation_func, compare_func, max_depth, max_expansion)
        self.selection_func = selection_func

    def search(self, root: TreeSearchNode, target: Callable[[TreeSearchNode], bool]) -> TreeSearchNode:
        node = root
        best_node = node
        depth = 0
        while depth < self.max_depth:
            if node.value is None:
                node.value = self.simulation_func(node)
            if target(node):
                return node
            if self.compare_func(best_node, node):
                best_node = node
            self.expand(node)
            if not node.children:
                break
            best_child = self.selection_func(node.children)
            node = best_child
            depth += 1
            if 0 < self.max_expansion <= self.expansion_count:
                break
        return best_node

    def expand(self, node: TreeSearchNode):
        for child_data in self.expansion_func(node):
            child = TreeSearchNode(child_data, parent=node)
            child.value = self.simulation_func(child)
            node.children.append(child)
            self.expansion_count += 1
            if 0 < self.max_expansion <= self.expansion_count:
                break


class MCTSSearchStrategy(SearchStrategy):
    strategy_name = "MCTS"

    def __init__(self,
         expansion_func: Callable[[TreeSearchNode], list[dict]],
         simulation_func: Callable[[TreeSearchNode], float],
         compare_func: Callable[[TreeSearchNode, TreeSearchNode], bool] = None,
         max_depth: int = 5,
         max_expansion: int = -1,
         exploration_constant: float = math.sqrt(2),
         ):
        super().__init__(expansion_func, simulation_func, compare_func, max_depth, max_expansion)
        self.exploration_constant = exploration_constant

    def search(self, root: TreeSearchNode, target: Callable[[TreeSearchNode], bool]) -> TreeSearchNode:
        while True:
            # 1. 选择一个未完全展开的节点
            node = self.select(root)
            # 2. 扩展节点
            self.expand(node)
            # 3. 模拟节点
            self.simulate(node)
            if target(node):
                return node
            # 4. 反向传播
            self.backpropagate(node, node.value)
            if 0 < self.max_expansion <= self.expansion_count:
                break
        return self.best_child(root, explore=False)

    def select(self, node: TreeSearchNode):
        """选择一个最高 UCT 值的子节点，知道找到一个未完全展开的节点"""
        while node.children:
            node = self.best_child(node, explore=True)
        return node

    def expand(self, node: TreeSearchNode):
        for child_data in self.expansion_func(node):
            child = TreeSearchNode(child_data, parent=node)
            node.children.append(child)
            self.expansion_count += 1
            if 0 < self.max_expansion <= self.expansion_count:
                break

    def simulate(self, node: TreeSearchNode):
        node.value = self.simulation_func(node)

    def backpropagate(self, node: TreeSearchNode, value: float):
        """反向传播，更新节点的值"""
        while node is not None:
            node.visit_count += 1
            node.mcts_value += value
            node = node.parent

    def best_child(self, node: TreeSearchNode, explore: bool = True):
        """选择最优子节点, 基于 UCT 公式"""
        if not node.children:
            return None

        if explore:
            # 使用 UCT（上限置信区间）公式选择最优子节点
            def uct_value(child: TreeSearchNode):
                if child.visit_count == 0:
                    return math.inf
                return (child.mcts_value / child.visit_count) + self.exploration_constant * math.sqrt(
                    math.log(node.visit_count) / child.visit_count
                )

            return max(node.children, key=uct_value)
        else:
            # 选择平均值最高的子节点
            return max(node.children, key=lambda n: (n.mcts_value / n.visit_count) if n.visit_count > 0 else -math.inf)


class TreeSearch:
    def __init__(
            self,
            root_state: dict,
            search_strategy: SearchStrategy,
    ):
        """ 初始化树搜索

        Args:
            root_state (dict): 根节点数据
            search_strategy (SearchStrategy): 搜索策略
        """
        self.root = TreeSearchNode(root_state)
        self.search_strategy = search_strategy

    def search(self, target: Callable[[TreeSearchNode], bool]) -> TreeSearchNode:
        return self.search_strategy.search(self.root, target)

    def print(self):
        """打印树"""
        print_str = ""
        def print_node(node, depth, prefix="", is_last=True):
            nonlocal print_str
            if depth > 0:
                branch = "|-- " if not is_last else "\\-- "
                node_data_str = json.dumps(node.data, ensure_ascii=False)
                print_str += f"{prefix}{branch}{node.value} ({node_data_str})\n"
            else:
                print_str += f"{node.value}\n"

            # 新前缀：保留竖线的部分，最后一个子节点无竖线
            if depth > 0:
                new_prefix = prefix + ("|   " if not is_last else "    ")
            else:
                new_prefix = ""
            for i, child in enumerate(node.children):
                if child.value is None:
                    continue
                print_node(child, depth + 1, new_prefix, i == len(node.children) - 1)

        print_node(self.root, 0)
        return print_str
