use std::cell::{RefCell, RefMut};
use std::rc::Rc;

/// An AVL tree node can have left and right children.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AVLTreeNodeSide {
    /// Represents the left child of a node.
    Left,
    /// Represents the right child of a node.
    Right,
}

/// A node in an AVL tree.
#[derive(Debug, Clone, PartialEq, Eq)] // 添加 PartialEq 和 Eq
pub struct AVLTreeNode<K, V> {
    children: [Option<Rc<RefCell<AVLTreeNode<K, V>>>>; 2],
    parent: Option<Rc<RefCell<AVLTreeNode<K, V>>>>, // 使用 Option 来表示可能为空的父节点
    key: K,
    value: V,
    height: i32,
}

impl<K, V> AVLTreeNode<K, V>
where
    K: PartialEq, // 确保 K 实现了 PartialEq
    V: PartialEq, // 确保 V 实现了 PartialEq
{
    /// Creates a new AVL tree node.
    pub fn new(key: K, value: V) -> Self {
        AVLTreeNode {
            children: [None, None],
            parent: None,
            key,
            value,
            height: 1,
        }
    }

    /// Retrieve the value at a given tree node.
    pub fn value(&self) -> &V {
        &self.value
    }

    /// Retrieve the key for a given tree node.
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Determines which side of its parent the given node is on.
    pub fn parent_side(&self) -> Option<AVLTreeNodeSide> {
        // Get the parent node as a reference
        let parent = self.parent.as_ref()?;

        // Iterate over the parent's children and find the position of `self`
        parent.borrow().children.iter().position(|child| {
            // Compare the child node with `self` by dereferencing the `Rc<RefCell<_>>`
            child.as_ref().map(|node| *node.borrow() == *self).unwrap_or(false)
        }).map(|side| if side == 0 { AVLTreeNodeSide::Left } else { AVLTreeNodeSide::Right })
    }

    /// Find the child of a given tree node.
    ///
    /// # Arguments
    ///
    /// * `side` - Which child node to get (left or right)
    ///
    /// # Returns
    ///
    /// The child of the tree node, or `None` if the node has no child on the given side.
    pub fn child(&self, side: AVLTreeNodeSide) -> Option<Rc<RefCell<AVLTreeNode<K, V>>>> {
        match side {
            AVLTreeNodeSide::Left => self.children[0].clone(),
            AVLTreeNodeSide::Right => self.children[1].clone(),
        }
    }

    /// Find the parent node of a given tree node.
    ///
    /// Returns the parent node of the tree node, or `None` if this is the root node.
    pub fn parent(&self) -> Option<Rc<RefCell<AVLTreeNode<K, V>>>> {
        self.parent.clone()
    }
}

impl<K, V> AVLTreeNode<K, V> {
    /// Find the height of a subtree.
    ///
    /// # Arguments
    ///
    /// * `node` - The root node of the subtree.
    ///
    /// # Returns
    ///
    /// The height of the subtree.
    pub fn subtree_height(&self) -> i32 {
        self.height
    }

    /// Update the height of the current node.
    ///
    /// This function calculates the height of the left and right subtrees and
    /// updates the height of the current node accordingly.
    pub fn update_height(&mut self) {
        let left_height = self.children[0].as_ref().map_or(0, |node| node.borrow().subtree_height());
        let right_height = self.children[1].as_ref().map_or(0, |node| node.borrow().subtree_height());
        self.height = std::cmp::max(left_height, right_height) + 1;
    }
}

/// An AVL tree balanced binary tree.
#[derive(Debug, Clone)]
pub struct AVLTree<K: Ord, V> {
    root_node: Option<Rc<RefCell<AVLTreeNode<K, V>>>>, // 使用 Option 来表示可能为空的根节点
    num_nodes: usize, // 树中节点的数量
}

impl<K: Ord + Clone + PartialEq, V: Clone + PartialEq> AVLTree<K, V> {
    /// Creates a new, empty AVL tree.
    pub fn new() -> Self {
        AVLTree {
            root_node: None,
            num_nodes: 0,
        }
    }

    /// Retrieve the number of entries in the tree.
    pub fn num_entries(&self) -> usize {
        self.num_nodes
    }

    /// Convert the keys in an AVL tree into a Rust `Vec`. This allows
    /// the tree to be used as an ordered set.
    ///
    /// # Returns
    ///
    /// A newly allocated `Vec` containing all the keys
    /// in the tree, in order. The length of the `Vec`
    /// is equal to the number of entries in the tree.
    pub fn to_vec(&self) -> Vec<K> {
        let mut array = Vec::with_capacity(self.num_nodes);
        let mut index = 0;
        self.add_subtree_to_vec(&self.root_node, &mut array, &mut index);
        array
    }

    /// Recursively adds the keys of the subtree to the `Vec` in an in-order traversal.
    fn add_subtree_to_vec(&self, subtree: &Option<Rc<RefCell<AVLTreeNode<K, V>>>>, array: &mut Vec<K>, index: &mut usize) {
        if let Some(node) = subtree {
            // Add left subtree first
            self.add_subtree_to_vec(&node.borrow().children[0], array, index);

            // Add this node
            array.push(node.borrow().key.clone());
            *index += 1;

            // Finally add right subtree
            self.add_subtree_to_vec(&node.borrow().children[1], array, index);
        }
    }

    /// Replaces a node in the AVL tree with another node.
    ///
    /// # Arguments
    ///
    /// * `node1` - The node to be replaced.
    /// * `node2` - The node to replace `node1`.
    pub fn replace_node(&mut self, node1: &mut AVLTreeNode<K, V>, mut node2: Option<Rc<RefCell<AVLTreeNode<K, V>>>>) {
        if let Some(ref mut node2) = node2 {
            node2.borrow_mut().parent = node1.parent.take();
        }

        if node1.parent.is_none() {
            self.root_node = node2;
        } else {
            let side = node1.parent_side().unwrap();
            if let Some(parent) = node1.parent.as_mut() {
                parent.borrow_mut().children[side as usize] = node2;
                parent.borrow_mut().update_height();
            }
        }
    }

    /// Find the root node of a tree.
    ///
    /// # Returns
    ///
    /// The root node of the tree, or `None` if the tree is empty.
    pub fn root_node(&self) -> Option<Rc<RefCell<AVLTreeNode<K, V>>>> {
        self.root_node.clone()
    }

    /// Search an AVL tree for a node with a particular key.
    /// This uses the tree as a mapping.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to search for.
    ///
    /// # Returns
    ///
    /// The tree node containing the given key, or `None` if no entry with the given key is found.
    pub fn lookup_node(&self, key: &K) -> Option<Rc<RefCell<AVLTreeNode<K, V>>>> {
        let mut node = self.root_node.clone();
        while let Some(current_node) = node {
            match key.cmp(&current_node.borrow().key) {
                std::cmp::Ordering::Equal => return Some(current_node.clone()),
                std::cmp::Ordering::Less => node = current_node.borrow().children[0].clone(),
                std::cmp::Ordering::Greater => node = current_node.borrow().children[1].clone(),
            }
        }
        None
    }

    /// Search an AVL tree for a value corresponding to a particular key.
    /// This uses the tree as a mapping. Note that this performs identically to
    /// `lookup_node`, except that the value at the node is returned rather than
    /// the node itself.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to search for.
    ///
    /// # Returns
    ///
    /// The value associated with the given key, or `None` if no entry with the
    /// given key is found.
    pub fn lookup(&self, key: &K) -> Option<V> {
        self.lookup_node(key).map(|node| node.borrow().value.clone())
    }

    /// Find a replacement node for the given node in the AVL tree.
    fn get_replacement_node(&self, node: &AVLTreeNode<K, V>) -> Option<Rc<RefCell<AVLTreeNode<K, V>>>> {
        let mut current = node.children[1].as_ref()?.clone();
        loop {
            let mut borrowed = current.borrow_mut();
            if let Some(left_child) = borrowed.children[0].clone() {
                drop(borrowed); // Drop the borrow before updating `current`
                current = left_child;
            } else {
                drop(borrowed); // Drop the borrow before returning
                return Some(current);
            }
        }
    }

    /// Balance the tree from the given node up to the root.
    fn balance_to_root(&mut self, node: Option<Rc<RefCell<AVLTreeNode<K, V>>>>) {
        let mut current = node;
        while let Some(node) = current {
            self.balance_node(&node);
            current = node.borrow().parent.clone();
        }
    }

    /// Balance a single node in the AVL tree.
    fn balance_node(&mut self, node: &Rc<RefCell<AVLTreeNode<K, V>>>) {
        // Implement the balancing logic here
    }

    /// Remove a node from the AVL tree.
    pub fn remove_node(&mut self, node: &mut AVLTreeNode<K, V>) {
        let swap_node = self.get_replacement_node(node);

        if let Some(swap_node) = swap_node {
            // Copy references from the node into the swap node
            for i in 0..2 {
                swap_node.borrow_mut().children[i] = node.children[i].take();
                if let Some(child) = &swap_node.borrow().children[i] {
                    child.borrow_mut().parent = Some(swap_node.clone());
                }
            }
            swap_node.borrow_mut().height = node.height;

            // Replace the node with the swap node
            self.replace_node(node, Some(swap_node.clone()));
        } else {
            // This is a leaf node, simply remove it
            self.replace_node(node, None);
        }

        // Update the number of nodes
        self.num_nodes -= 1;

        // Rebalance the tree
        self.balance_to_root(node.parent.clone());
    }

    /// Remove an entry from the AVL tree by key.
    pub fn remove(&mut self, key: &K) -> bool {
        if let Some(node) = self.lookup_node(key) {
            self.remove_node(&mut node.borrow_mut());
            true
        } else {
            false
        }
    }

    /// Inserts a new key-value pair into the AVL tree.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to insert.
    /// * `value` - The value to insert.
    ///
    /// # Returns
    ///
    /// The newly created tree node containing the key and value, or `None` if it was not possible
    /// to allocate the new memory.
    pub fn insert(&mut self, key: K, value: V) -> Option<Rc<RefCell<AVLTreeNode<K, V>>>> {
        let mut rover = &mut self.root_node;
        let mut previous_node = None;

        // Walk down the tree until we reach a NULL pointer
        while let Some(node) = rover {
            previous_node = Some(node.clone());
            if key < node.borrow().key {
                rover = &mut node.borrow_mut().children[0];
            } else {
                rover = &mut node.borrow_mut().children[1];
            }
        }

        // Create a new node. Use the last node visited as the parent link.
        let new_node = Rc::new(RefCell::new(AVLTreeNode::new(key, value)));
        if let Some(parent) = previous_node {
            new_node.borrow_mut().parent = Some(parent.clone());
            if key < parent.borrow().key {
                parent.borrow_mut().children[0] = Some(new_node.clone());
            } else {
                parent.borrow_mut().children[1] = Some(new_node.clone());
            }
        } else {
            self.root_node = Some(new_node.clone());
        }

        // Rebalance the tree, starting from the previous node.
        self.balance_to_root(previous_node);

        // Keep track of the number of entries
        self.num_nodes += 1;

        Some(new_node)
    }
}

impl<K, V> Drop for AVLTreeNode<K, V> {
    fn drop(&mut self) {
        if let Some(left_child) = self.children[0].take() {
            drop(left_child);
        }
        if let Some(right_child) = self.children[1].take() {
            drop(right_child);
        }
    }
}

impl<K: Ord, V> Drop for AVLTree<K, V> {
    fn drop(&mut self) {
        if let Some(root_node) = self.root_node.take() {
            drop(root_node);
        }
    }
}