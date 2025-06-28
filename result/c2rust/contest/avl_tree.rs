use ::libc;
extern "C" {
    fn alloc_test_malloc(bytes: size_t) -> *mut libc::c_void;
    fn alloc_test_free(ptr: *mut libc::c_void);
}
pub type size_t = libc::c_ulong;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _AVLTree {
    pub root_node: *mut AVLTreeNode,
    pub compare_func: AVLTreeCompareFunc,
    pub num_nodes: libc::c_uint,
}
pub type AVLTreeCompareFunc = Option::<
    unsafe extern "C" fn(AVLTreeValue, AVLTreeValue) -> libc::c_int,
>;
pub type AVLTreeValue = *mut libc::c_void;
pub type AVLTreeNode = _AVLTreeNode;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _AVLTreeNode {
    pub children: [*mut AVLTreeNode; 2],
    pub parent: *mut AVLTreeNode,
    pub key: AVLTreeKey,
    pub value: AVLTreeValue,
    pub height: libc::c_int,
}
pub type AVLTreeKey = *mut libc::c_void;
pub type AVLTree = _AVLTree;
pub type AVLTreeNodeSide = libc::c_uint;
pub const AVL_TREE_NODE_RIGHT: AVLTreeNodeSide = 1;
pub const AVL_TREE_NODE_LEFT: AVLTreeNodeSide = 0;
#[no_mangle]
pub unsafe extern "C" fn avl_tree_new(
    mut compare_func: AVLTreeCompareFunc,
) -> *mut AVLTree {
    let mut new_tree: *mut AVLTree = 0 as *mut AVLTree;
    new_tree = alloc_test_malloc(::core::mem::size_of::<AVLTree>() as libc::c_ulong)
        as *mut AVLTree;
    if new_tree.is_null() {
        return 0 as *mut AVLTree;
    }
    (*new_tree).root_node = 0 as *mut AVLTreeNode;
    (*new_tree).compare_func = compare_func;
    (*new_tree).num_nodes = 0 as libc::c_int as libc::c_uint;
    return new_tree;
}
unsafe extern "C" fn avl_tree_free_subtree(
    mut tree: *mut AVLTree,
    mut node: *mut AVLTreeNode,
) {
    if node.is_null() {
        return;
    }
    avl_tree_free_subtree(
        tree,
        (*node).children[AVL_TREE_NODE_LEFT as libc::c_int as usize],
    );
    avl_tree_free_subtree(
        tree,
        (*node).children[AVL_TREE_NODE_RIGHT as libc::c_int as usize],
    );
    alloc_test_free(node as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn avl_tree_free(mut tree: *mut AVLTree) {
    avl_tree_free_subtree(tree, (*tree).root_node);
    alloc_test_free(tree as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn avl_tree_subtree_height(
    mut node: *mut AVLTreeNode,
) -> libc::c_int {
    if node.is_null() { return 0 as libc::c_int } else { return (*node).height };
}
unsafe extern "C" fn avl_tree_update_height(mut node: *mut AVLTreeNode) {
    let mut left_subtree: *mut AVLTreeNode = 0 as *mut AVLTreeNode;
    let mut right_subtree: *mut AVLTreeNode = 0 as *mut AVLTreeNode;
    let mut left_height: libc::c_int = 0;
    let mut right_height: libc::c_int = 0;
    left_subtree = (*node).children[AVL_TREE_NODE_LEFT as libc::c_int as usize];
    right_subtree = (*node).children[AVL_TREE_NODE_RIGHT as libc::c_int as usize];
    left_height = avl_tree_subtree_height(left_subtree);
    right_height = avl_tree_subtree_height(right_subtree);
    if left_height > right_height {
        (*node).height = left_height + 1 as libc::c_int;
    } else {
        (*node).height = right_height + 1 as libc::c_int;
    };
}
unsafe extern "C" fn avl_tree_node_parent_side(
    mut node: *mut AVLTreeNode,
) -> AVLTreeNodeSide {
    if (*(*node).parent).children[AVL_TREE_NODE_LEFT as libc::c_int as usize] == node {
        return AVL_TREE_NODE_LEFT
    } else {
        return AVL_TREE_NODE_RIGHT
    };
}
unsafe extern "C" fn avl_tree_node_replace(
    mut tree: *mut AVLTree,
    mut node1: *mut AVLTreeNode,
    mut node2: *mut AVLTreeNode,
) {
    let mut side: libc::c_int = 0;
    if !node2.is_null() {
        (*node2).parent = (*node1).parent;
    }
    if ((*node1).parent).is_null() {
        (*tree).root_node = node2;
    } else {
        side = avl_tree_node_parent_side(node1) as libc::c_int;
        (*(*node1).parent).children[side as usize] = node2;
        avl_tree_update_height((*node1).parent);
    };
}
unsafe extern "C" fn avl_tree_rotate(
    mut tree: *mut AVLTree,
    mut node: *mut AVLTreeNode,
    mut direction: AVLTreeNodeSide,
) -> *mut AVLTreeNode {
    let mut new_root: *mut AVLTreeNode = 0 as *mut AVLTreeNode;
    new_root = (*node)
        .children[(1 as libc::c_int as libc::c_uint)
        .wrapping_sub(direction as libc::c_uint) as usize];
    avl_tree_node_replace(tree, node, new_root);
    (*node)
        .children[(1 as libc::c_int as libc::c_uint)
        .wrapping_sub(direction as libc::c_uint)
        as usize] = (*new_root).children[direction as usize];
    (*new_root).children[direction as usize] = node;
    (*node).parent = new_root;
    if !((*node)
        .children[(1 as libc::c_int as libc::c_uint)
        .wrapping_sub(direction as libc::c_uint) as usize])
        .is_null()
    {
        (*(*node)
            .children[(1 as libc::c_int as libc::c_uint)
            .wrapping_sub(direction as libc::c_uint) as usize])
            .parent = node;
    }
    avl_tree_update_height(new_root);
    avl_tree_update_height(node);
    return new_root;
}
unsafe extern "C" fn avl_tree_node_balance(
    mut tree: *mut AVLTree,
    mut node: *mut AVLTreeNode,
) -> *mut AVLTreeNode {
    let mut left_subtree: *mut AVLTreeNode = 0 as *mut AVLTreeNode;
    let mut right_subtree: *mut AVLTreeNode = 0 as *mut AVLTreeNode;
    let mut child: *mut AVLTreeNode = 0 as *mut AVLTreeNode;
    let mut diff: libc::c_int = 0;
    left_subtree = (*node).children[AVL_TREE_NODE_LEFT as libc::c_int as usize];
    right_subtree = (*node).children[AVL_TREE_NODE_RIGHT as libc::c_int as usize];
    diff = avl_tree_subtree_height(right_subtree)
        - avl_tree_subtree_height(left_subtree);
    if diff >= 2 as libc::c_int {
        child = right_subtree;
        if avl_tree_subtree_height(
            (*child).children[AVL_TREE_NODE_RIGHT as libc::c_int as usize],
        )
            < avl_tree_subtree_height(
                (*child).children[AVL_TREE_NODE_LEFT as libc::c_int as usize],
            )
        {
            avl_tree_rotate(tree, right_subtree, AVL_TREE_NODE_RIGHT);
        }
        node = avl_tree_rotate(tree, node, AVL_TREE_NODE_LEFT);
    } else if diff <= -(2 as libc::c_int) {
        child = (*node).children[AVL_TREE_NODE_LEFT as libc::c_int as usize];
        if avl_tree_subtree_height(
            (*child).children[AVL_TREE_NODE_LEFT as libc::c_int as usize],
        )
            < avl_tree_subtree_height(
                (*child).children[AVL_TREE_NODE_RIGHT as libc::c_int as usize],
            )
        {
            avl_tree_rotate(tree, left_subtree, AVL_TREE_NODE_LEFT);
        }
        node = avl_tree_rotate(tree, node, AVL_TREE_NODE_RIGHT);
    }
    avl_tree_update_height(node);
    return node;
}
unsafe extern "C" fn avl_tree_balance_to_root(
    mut tree: *mut AVLTree,
    mut node: *mut AVLTreeNode,
) {
    let mut rover: *mut AVLTreeNode = 0 as *mut AVLTreeNode;
    rover = node;
    while !rover.is_null() {
        rover = avl_tree_node_balance(tree, rover);
        rover = (*rover).parent;
    }
}
#[no_mangle]
pub unsafe extern "C" fn avl_tree_insert(
    mut tree: *mut AVLTree,
    mut key: AVLTreeKey,
    mut value: AVLTreeValue,
) -> *mut AVLTreeNode {
    let mut rover: *mut *mut AVLTreeNode = 0 as *mut *mut AVLTreeNode;
    let mut new_node: *mut AVLTreeNode = 0 as *mut AVLTreeNode;
    let mut previous_node: *mut AVLTreeNode = 0 as *mut AVLTreeNode;
    rover = &mut (*tree).root_node;
    previous_node = 0 as *mut AVLTreeNode;
    while !(*rover).is_null() {
        previous_node = *rover;
        if ((*tree).compare_func).expect("non-null function pointer")(key, (**rover).key)
            < 0 as libc::c_int
        {
            rover = &mut *((**rover).children)
                .as_mut_ptr()
                .offset(AVL_TREE_NODE_LEFT as libc::c_int as isize)
                as *mut *mut AVLTreeNode;
        } else {
            rover = &mut *((**rover).children)
                .as_mut_ptr()
                .offset(AVL_TREE_NODE_RIGHT as libc::c_int as isize)
                as *mut *mut AVLTreeNode;
        }
    }
    new_node = alloc_test_malloc(::core::mem::size_of::<AVLTreeNode>() as libc::c_ulong)
        as *mut AVLTreeNode;
    if new_node.is_null() {
        return 0 as *mut AVLTreeNode;
    }
    (*new_node)
        .children[AVL_TREE_NODE_LEFT as libc::c_int as usize] = 0 as *mut AVLTreeNode;
    (*new_node)
        .children[AVL_TREE_NODE_RIGHT as libc::c_int as usize] = 0 as *mut AVLTreeNode;
    (*new_node).parent = previous_node;
    (*new_node).key = key;
    (*new_node).value = value;
    (*new_node).height = 1 as libc::c_int;
    *rover = new_node;
    avl_tree_balance_to_root(tree, previous_node);
    (*tree).num_nodes = ((*tree).num_nodes).wrapping_add(1);
    (*tree).num_nodes;
    return new_node;
}
unsafe extern "C" fn avl_tree_node_get_replacement(
    mut tree: *mut AVLTree,
    mut node: *mut AVLTreeNode,
) -> *mut AVLTreeNode {
    let mut left_subtree: *mut AVLTreeNode = 0 as *mut AVLTreeNode;
    let mut right_subtree: *mut AVLTreeNode = 0 as *mut AVLTreeNode;
    let mut result: *mut AVLTreeNode = 0 as *mut AVLTreeNode;
    let mut child: *mut AVLTreeNode = 0 as *mut AVLTreeNode;
    let mut left_height: libc::c_int = 0;
    let mut right_height: libc::c_int = 0;
    let mut side: libc::c_int = 0;
    left_subtree = (*node).children[AVL_TREE_NODE_LEFT as libc::c_int as usize];
    right_subtree = (*node).children[AVL_TREE_NODE_RIGHT as libc::c_int as usize];
    if left_subtree.is_null() && right_subtree.is_null() {
        return 0 as *mut AVLTreeNode;
    }
    left_height = avl_tree_subtree_height(left_subtree);
    right_height = avl_tree_subtree_height(right_subtree);
    if left_height < right_height {
        side = AVL_TREE_NODE_RIGHT as libc::c_int;
    } else {
        side = AVL_TREE_NODE_LEFT as libc::c_int;
    }
    result = (*node).children[side as usize];
    while !((*result).children[(1 as libc::c_int - side) as usize]).is_null() {
        result = (*result).children[(1 as libc::c_int - side) as usize];
    }
    child = (*result).children[side as usize];
    avl_tree_node_replace(tree, result, child);
    avl_tree_update_height((*result).parent);
    return result;
}
#[no_mangle]
pub unsafe extern "C" fn avl_tree_remove_node(
    mut tree: *mut AVLTree,
    mut node: *mut AVLTreeNode,
) {
    let mut swap_node: *mut AVLTreeNode = 0 as *mut AVLTreeNode;
    let mut balance_startpoint: *mut AVLTreeNode = 0 as *mut AVLTreeNode;
    let mut i: libc::c_int = 0;
    swap_node = avl_tree_node_get_replacement(tree, node);
    if swap_node.is_null() {
        avl_tree_node_replace(tree, node, 0 as *mut AVLTreeNode);
        balance_startpoint = (*node).parent;
    } else {
        if (*swap_node).parent == node {
            balance_startpoint = swap_node;
        } else {
            balance_startpoint = (*swap_node).parent;
        }
        i = 0 as libc::c_int;
        while i < 2 as libc::c_int {
            (*swap_node).children[i as usize] = (*node).children[i as usize];
            if !((*swap_node).children[i as usize]).is_null() {
                (*(*swap_node).children[i as usize]).parent = swap_node;
            }
            i += 1;
            i;
        }
        (*swap_node).height = (*node).height;
        avl_tree_node_replace(tree, node, swap_node);
    }
    alloc_test_free(node as *mut libc::c_void);
    (*tree).num_nodes = ((*tree).num_nodes).wrapping_sub(1);
    (*tree).num_nodes;
    avl_tree_balance_to_root(tree, balance_startpoint);
}
#[no_mangle]
pub unsafe extern "C" fn avl_tree_remove(
    mut tree: *mut AVLTree,
    mut key: AVLTreeKey,
) -> libc::c_int {
    let mut node: *mut AVLTreeNode = 0 as *mut AVLTreeNode;
    node = avl_tree_lookup_node(tree, key);
    if node.is_null() {
        return 0 as libc::c_int;
    }
    avl_tree_remove_node(tree, node);
    return 1 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn avl_tree_lookup_node(
    mut tree: *mut AVLTree,
    mut key: AVLTreeKey,
) -> *mut AVLTreeNode {
    let mut node: *mut AVLTreeNode = 0 as *mut AVLTreeNode;
    let mut diff: libc::c_int = 0;
    node = (*tree).root_node;
    while !node.is_null() {
        diff = ((*tree).compare_func)
            .expect("non-null function pointer")(key, (*node).key);
        if diff == 0 as libc::c_int {
            return node
        } else if diff < 0 as libc::c_int {
            node = (*node).children[AVL_TREE_NODE_LEFT as libc::c_int as usize];
        } else {
            node = (*node).children[AVL_TREE_NODE_RIGHT as libc::c_int as usize];
        }
    }
    return 0 as *mut AVLTreeNode;
}
#[no_mangle]
pub unsafe extern "C" fn avl_tree_lookup(
    mut tree: *mut AVLTree,
    mut key: AVLTreeKey,
) -> AVLTreeValue {
    let mut node: *mut AVLTreeNode = 0 as *mut AVLTreeNode;
    node = avl_tree_lookup_node(tree, key);
    if node.is_null() { return 0 as *mut libc::c_void } else { return (*node).value };
}
#[no_mangle]
pub unsafe extern "C" fn avl_tree_root_node(mut tree: *mut AVLTree) -> *mut AVLTreeNode {
    return (*tree).root_node;
}
#[no_mangle]
pub unsafe extern "C" fn avl_tree_node_key(mut node: *mut AVLTreeNode) -> AVLTreeKey {
    return (*node).key;
}
#[no_mangle]
pub unsafe extern "C" fn avl_tree_node_value(
    mut node: *mut AVLTreeNode,
) -> AVLTreeValue {
    return (*node).value;
}
#[no_mangle]
pub unsafe extern "C" fn avl_tree_node_child(
    mut node: *mut AVLTreeNode,
    mut side: AVLTreeNodeSide,
) -> *mut AVLTreeNode {
    if side as libc::c_uint == AVL_TREE_NODE_LEFT as libc::c_int as libc::c_uint
        || side as libc::c_uint == AVL_TREE_NODE_RIGHT as libc::c_int as libc::c_uint
    {
        return (*node).children[side as usize]
    } else {
        return 0 as *mut AVLTreeNode
    };
}
#[no_mangle]
pub unsafe extern "C" fn avl_tree_node_parent(
    mut node: *mut AVLTreeNode,
) -> *mut AVLTreeNode {
    return (*node).parent;
}
#[no_mangle]
pub unsafe extern "C" fn avl_tree_num_entries(mut tree: *mut AVLTree) -> libc::c_uint {
    return (*tree).num_nodes;
}
unsafe extern "C" fn avl_tree_to_array_add_subtree(
    mut subtree: *mut AVLTreeNode,
    mut array: *mut AVLTreeValue,
    mut index: *mut libc::c_int,
) {
    if subtree.is_null() {
        return;
    }
    avl_tree_to_array_add_subtree(
        (*subtree).children[AVL_TREE_NODE_LEFT as libc::c_int as usize],
        array,
        index,
    );
    let ref mut fresh0 = *array.offset(*index as isize);
    *fresh0 = (*subtree).key;
    *index += 1;
    *index;
    avl_tree_to_array_add_subtree(
        (*subtree).children[AVL_TREE_NODE_RIGHT as libc::c_int as usize],
        array,
        index,
    );
}
#[no_mangle]
pub unsafe extern "C" fn avl_tree_to_array(mut tree: *mut AVLTree) -> *mut AVLTreeValue {
    let mut array: *mut AVLTreeValue = 0 as *mut AVLTreeValue;
    let mut index: libc::c_int = 0;
    array = alloc_test_malloc(
        (::core::mem::size_of::<AVLTreeValue>() as libc::c_ulong)
            .wrapping_mul((*tree).num_nodes as libc::c_ulong),
    ) as *mut AVLTreeValue;
    if array.is_null() {
        return 0 as *mut AVLTreeValue;
    }
    index = 0 as libc::c_int;
    avl_tree_to_array_add_subtree((*tree).root_node, array, &mut index);
    return array;
}
