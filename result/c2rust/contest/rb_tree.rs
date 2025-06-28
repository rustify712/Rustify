use ::libc;
extern "C" {
    fn alloc_test_malloc(bytes: size_t) -> *mut libc::c_void;
    fn alloc_test_free(ptr: *mut libc::c_void);
}
pub type size_t = libc::c_ulong;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _RBTree {
    pub root_node: *mut RBTreeNode,
    pub compare_func: RBTreeCompareFunc,
    pub num_nodes: libc::c_int,
}
pub type RBTreeCompareFunc = Option::<
    unsafe extern "C" fn(RBTreeValue, RBTreeValue) -> libc::c_int,
>;
pub type RBTreeValue = *mut libc::c_void;
pub type RBTreeNode = _RBTreeNode;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _RBTreeNode {
    pub color: RBTreeNodeColor,
    pub key: RBTreeKey,
    pub value: RBTreeValue,
    pub parent: *mut RBTreeNode,
    pub children: [*mut RBTreeNode; 2],
}
pub type RBTreeKey = *mut libc::c_void;
pub type RBTreeNodeColor = libc::c_uint;
pub const RB_TREE_NODE_BLACK: RBTreeNodeColor = 1;
pub const RB_TREE_NODE_RED: RBTreeNodeColor = 0;
pub type RBTree = _RBTree;
pub type RBTreeNodeSide = libc::c_uint;
pub const RB_TREE_NODE_RIGHT: RBTreeNodeSide = 1;
pub const RB_TREE_NODE_LEFT: RBTreeNodeSide = 0;
unsafe extern "C" fn rb_tree_node_side(mut node: *mut RBTreeNode) -> RBTreeNodeSide {
    if (*(*node).parent).children[RB_TREE_NODE_LEFT as libc::c_int as usize] == node {
        return RB_TREE_NODE_LEFT
    } else {
        return RB_TREE_NODE_RIGHT
    };
}
unsafe extern "C" fn rb_tree_node_sibling(mut node: *mut RBTreeNode) -> *mut RBTreeNode {
    let mut side: RBTreeNodeSide = RB_TREE_NODE_LEFT;
    side = rb_tree_node_side(node);
    return (*(*node).parent)
        .children[(1 as libc::c_int as libc::c_uint).wrapping_sub(side as libc::c_uint)
        as usize];
}
#[no_mangle]
pub unsafe extern "C" fn rb_tree_node_uncle(
    mut node: *mut RBTreeNode,
) -> *mut RBTreeNode {
    return rb_tree_node_sibling((*node).parent);
}
unsafe extern "C" fn rb_tree_node_replace(
    mut tree: *mut RBTree,
    mut node1: *mut RBTreeNode,
    mut node2: *mut RBTreeNode,
) {
    let mut side: libc::c_int = 0;
    if !node2.is_null() {
        (*node2).parent = (*node1).parent;
    }
    if ((*node1).parent).is_null() {
        (*tree).root_node = node2;
    } else {
        side = rb_tree_node_side(node1) as libc::c_int;
        (*(*node1).parent).children[side as usize] = node2;
    };
}
unsafe extern "C" fn rb_tree_rotate(
    mut tree: *mut RBTree,
    mut node: *mut RBTreeNode,
    mut direction: RBTreeNodeSide,
) -> *mut RBTreeNode {
    let mut new_root: *mut RBTreeNode = 0 as *mut RBTreeNode;
    new_root = (*node)
        .children[(1 as libc::c_int as libc::c_uint)
        .wrapping_sub(direction as libc::c_uint) as usize];
    rb_tree_node_replace(tree, node, new_root);
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
    return new_root;
}
#[no_mangle]
pub unsafe extern "C" fn rb_tree_new(
    mut compare_func: RBTreeCompareFunc,
) -> *mut RBTree {
    let mut new_tree: *mut RBTree = 0 as *mut RBTree;
    new_tree = alloc_test_malloc(::core::mem::size_of::<RBTree>() as libc::c_ulong)
        as *mut RBTree;
    if new_tree.is_null() {
        return 0 as *mut RBTree;
    }
    (*new_tree).root_node = 0 as *mut RBTreeNode;
    (*new_tree).num_nodes = 0 as libc::c_int;
    (*new_tree).compare_func = compare_func;
    return new_tree;
}
unsafe extern "C" fn rb_tree_free_subtree(mut node: *mut RBTreeNode) {
    if !node.is_null() {
        rb_tree_free_subtree(
            (*node).children[RB_TREE_NODE_LEFT as libc::c_int as usize],
        );
        rb_tree_free_subtree(
            (*node).children[RB_TREE_NODE_RIGHT as libc::c_int as usize],
        );
        alloc_test_free(node as *mut libc::c_void);
    }
}
#[no_mangle]
pub unsafe extern "C" fn rb_tree_free(mut tree: *mut RBTree) {
    rb_tree_free_subtree((*tree).root_node);
    alloc_test_free(tree as *mut libc::c_void);
}
unsafe extern "C" fn rb_tree_insert_case1(
    mut tree: *mut RBTree,
    mut node: *mut RBTreeNode,
) {
    if ((*node).parent).is_null() {
        (*node).color = RB_TREE_NODE_BLACK;
    } else {
        rb_tree_insert_case2(tree, node);
    };
}
unsafe extern "C" fn rb_tree_insert_case2(
    mut tree: *mut RBTree,
    mut node: *mut RBTreeNode,
) {
    if (*(*node).parent).color as libc::c_uint
        != RB_TREE_NODE_BLACK as libc::c_int as libc::c_uint
    {
        rb_tree_insert_case3(tree, node);
    }
}
unsafe extern "C" fn rb_tree_insert_case3(
    mut tree: *mut RBTree,
    mut node: *mut RBTreeNode,
) {
    let mut grandparent: *mut RBTreeNode = 0 as *mut RBTreeNode;
    let mut uncle: *mut RBTreeNode = 0 as *mut RBTreeNode;
    grandparent = (*(*node).parent).parent;
    uncle = rb_tree_node_uncle(node);
    if !uncle.is_null()
        && (*uncle).color as libc::c_uint
            == RB_TREE_NODE_RED as libc::c_int as libc::c_uint
    {
        (*(*node).parent).color = RB_TREE_NODE_BLACK;
        (*uncle).color = RB_TREE_NODE_BLACK;
        (*grandparent).color = RB_TREE_NODE_RED;
        rb_tree_insert_case1(tree, grandparent);
    } else {
        rb_tree_insert_case4(tree, node);
    };
}
unsafe extern "C" fn rb_tree_insert_case4(
    mut tree: *mut RBTree,
    mut node: *mut RBTreeNode,
) {
    let mut next_node: *mut RBTreeNode = 0 as *mut RBTreeNode;
    let mut side: RBTreeNodeSide = RB_TREE_NODE_LEFT;
    side = rb_tree_node_side(node);
    if side as libc::c_uint != rb_tree_node_side((*node).parent) as libc::c_uint {
        next_node = (*node).parent;
        rb_tree_rotate(
            tree,
            (*node).parent,
            (1 as libc::c_int as libc::c_uint).wrapping_sub(side as libc::c_uint)
                as RBTreeNodeSide,
        );
    } else {
        next_node = node;
    }
    rb_tree_insert_case5(tree, next_node);
}
unsafe extern "C" fn rb_tree_insert_case5(
    mut tree: *mut RBTree,
    mut node: *mut RBTreeNode,
) {
    let mut parent: *mut RBTreeNode = 0 as *mut RBTreeNode;
    let mut grandparent: *mut RBTreeNode = 0 as *mut RBTreeNode;
    let mut side: RBTreeNodeSide = RB_TREE_NODE_LEFT;
    parent = (*node).parent;
    grandparent = (*parent).parent;
    side = rb_tree_node_side(node);
    rb_tree_rotate(
        tree,
        grandparent,
        (1 as libc::c_int as libc::c_uint).wrapping_sub(side as libc::c_uint)
            as RBTreeNodeSide,
    );
    (*parent).color = RB_TREE_NODE_BLACK;
    (*grandparent).color = RB_TREE_NODE_RED;
}
#[no_mangle]
pub unsafe extern "C" fn rb_tree_insert(
    mut tree: *mut RBTree,
    mut key: RBTreeKey,
    mut value: RBTreeValue,
) -> *mut RBTreeNode {
    let mut node: *mut RBTreeNode = 0 as *mut RBTreeNode;
    let mut rover: *mut *mut RBTreeNode = 0 as *mut *mut RBTreeNode;
    let mut parent: *mut RBTreeNode = 0 as *mut RBTreeNode;
    let mut side: RBTreeNodeSide = RB_TREE_NODE_LEFT;
    node = alloc_test_malloc(::core::mem::size_of::<RBTreeNode>() as libc::c_ulong)
        as *mut RBTreeNode;
    if node.is_null() {
        return 0 as *mut RBTreeNode;
    }
    (*node).key = key;
    (*node).value = value;
    (*node).color = RB_TREE_NODE_RED;
    (*node).children[RB_TREE_NODE_LEFT as libc::c_int as usize] = 0 as *mut RBTreeNode;
    (*node).children[RB_TREE_NODE_RIGHT as libc::c_int as usize] = 0 as *mut RBTreeNode;
    parent = 0 as *mut RBTreeNode;
    rover = &mut (*tree).root_node;
    while !(*rover).is_null() {
        parent = *rover;
        if ((*tree).compare_func)
            .expect("non-null function pointer")(value, (**rover).value)
            < 0 as libc::c_int
        {
            side = RB_TREE_NODE_LEFT;
        } else {
            side = RB_TREE_NODE_RIGHT;
        }
        rover = &mut *((**rover).children).as_mut_ptr().offset(side as isize)
            as *mut *mut RBTreeNode;
    }
    *rover = node;
    (*node).parent = parent;
    rb_tree_insert_case1(tree, node);
    (*tree).num_nodes += 1;
    (*tree).num_nodes;
    return node;
}
#[no_mangle]
pub unsafe extern "C" fn rb_tree_lookup_node(
    mut tree: *mut RBTree,
    mut key: RBTreeKey,
) -> *mut RBTreeNode {
    let mut node: *mut RBTreeNode = 0 as *mut RBTreeNode;
    let mut side: RBTreeNodeSide = RB_TREE_NODE_LEFT;
    let mut diff: libc::c_int = 0;
    node = (*tree).root_node;
    while !node.is_null() {
        diff = ((*tree).compare_func)
            .expect("non-null function pointer")(key, (*node).key);
        if diff == 0 as libc::c_int {
            return node
        } else if diff < 0 as libc::c_int {
            side = RB_TREE_NODE_LEFT;
        } else {
            side = RB_TREE_NODE_RIGHT;
        }
        node = (*node).children[side as usize];
    }
    return 0 as *mut RBTreeNode;
}
#[no_mangle]
pub unsafe extern "C" fn rb_tree_lookup(
    mut tree: *mut RBTree,
    mut key: RBTreeKey,
) -> RBTreeValue {
    let mut node: *mut RBTreeNode = 0 as *mut RBTreeNode;
    node = rb_tree_lookup_node(tree, key);
    if node.is_null() { return 0 as *mut libc::c_void } else { return (*node).value };
}
#[no_mangle]
pub unsafe extern "C" fn rb_tree_remove_node(
    mut tree: *mut RBTree,
    mut node: *mut RBTreeNode,
) {}
#[no_mangle]
pub unsafe extern "C" fn rb_tree_remove(
    mut tree: *mut RBTree,
    mut key: RBTreeKey,
) -> libc::c_int {
    let mut node: *mut RBTreeNode = 0 as *mut RBTreeNode;
    node = rb_tree_lookup_node(tree, key);
    if node.is_null() {
        return 0 as libc::c_int;
    }
    rb_tree_remove_node(tree, node);
    return 1 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn rb_tree_root_node(mut tree: *mut RBTree) -> *mut RBTreeNode {
    return (*tree).root_node;
}
#[no_mangle]
pub unsafe extern "C" fn rb_tree_node_key(mut node: *mut RBTreeNode) -> RBTreeKey {
    return (*node).key;
}
#[no_mangle]
pub unsafe extern "C" fn rb_tree_node_value(mut node: *mut RBTreeNode) -> RBTreeValue {
    return (*node).value;
}
#[no_mangle]
pub unsafe extern "C" fn rb_tree_node_child(
    mut node: *mut RBTreeNode,
    mut side: RBTreeNodeSide,
) -> *mut RBTreeNode {
    if side as libc::c_uint == RB_TREE_NODE_LEFT as libc::c_int as libc::c_uint
        || side as libc::c_uint == RB_TREE_NODE_RIGHT as libc::c_int as libc::c_uint
    {
        return (*node).children[side as usize]
    } else {
        return 0 as *mut RBTreeNode
    };
}
#[no_mangle]
pub unsafe extern "C" fn rb_tree_node_parent(
    mut node: *mut RBTreeNode,
) -> *mut RBTreeNode {
    return (*node).parent;
}
#[no_mangle]
pub unsafe extern "C" fn rb_tree_to_array(mut tree: *mut RBTree) -> *mut RBTreeValue {
    return 0 as *mut RBTreeValue;
}
#[no_mangle]
pub unsafe extern "C" fn rb_tree_num_entries(mut tree: *mut RBTree) -> libc::c_int {
    return (*tree).num_nodes;
}
