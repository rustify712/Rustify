/// Each node in a red-black tree is either red or black.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RBTreeNodeColor {
    Red,
    Black,
}

/// A node in a red-black tree can have left and right children.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RBTreeNodeSide {
    Left,
    Right,
}

/// A node in a red-black tree.
#[derive(Debug, Clone, PartialEq)]
pub struct RBTreeNode<K, V> {
    pub color: RBTreeNodeColor,
    pub key: K,
    pub value: V,
    pub parent: Option<Box<RBTreeNode<K, V>>>, // Parent node
    pub children: [Option<Box<RBTreeNode<K, V>>>; 2], // Left and right children
}

impl<K: PartialEq, V: PartialEq> RBTreeNode<K, V> {
    /// Creates a new node with the given key, value, and color.
    pub fn new(key: K, value: V, color: RBTreeNodeColor) -> Self {
        RBTreeNode {
            color,
            key,
            value,
            parent: None,
            children: [None, None],
        }
    }

    /// Retrieve the key for a given tree node.
    pub fn key(&self) -> &K {
        &self.key
    }

    /// Retrieve the value at a given tree node.
    pub fn value(&self) -> &V {
        &self.value
    }

    /// Find the parent node of a given tree node.
    ///
    /// # Returns
    /// The parent node of the tree node, or `None` if this is the root node.
    pub fn parent(&self) -> Option<&RBTreeNode<K, V>> {
        self.parent.as_deref()
    }

    /// Recursively frees the subtree rooted at the given node.
    pub fn free_subtree(node: Option<Box<RBTreeNode<K, V>>>) {
        if let Some(mut boxed_node) = node {
            // Recurse to subnodes
            Self::free_subtree(boxed_node.children[0].take());
            Self::free_subtree(boxed_node.children[1].take());
            // The node will be automatically dropped when it goes out of scope
        }
    }

    /// Get a child of a given tree node.
    ///
    /// # Arguments
    ///
    /// * `side` - The side relative to the node.
    ///
    /// # Returns
    ///
    /// The child of the tree node, or `None` if the node has no child on the specified side.
    pub fn child(&self, side: RBTreeNodeSide) -> Option<&RBTreeNode<K, V>> {
        match side {
            RBTreeNodeSide::Left => self.children[0].as_deref(),
            RBTreeNodeSide::Right => self.children[1].as_deref(),
        }
    }

    /// Determines whether the current node is a left or right child of its parent.
    ///
    /// # Returns
    /// The side of the node relative to its parent, or `None` if the node is the root.
    pub fn side(&self) -> Option<RBTreeNodeSide> {
        if let Some(parent) = self.parent() {
            if parent.children[0].as_deref() == Some(self) {
                Some(RBTreeNodeSide::Left)
            } else {
                Some(RBTreeNodeSide::Right)
            }
        } else {
            None
        }
    }

    /// Finds the sibling of the current node.
    pub fn sibling(&self) -> Option<&RBTreeNode<K, V>> {
        if let Some(parent) = self.parent() {
            match self.side() {
                Some(RBTreeNodeSide::Left) => parent.children[1].as_deref(),
                Some(RBTreeNodeSide::Right) => parent.children[0].as_deref(),
                None => None,
            }
        } else {
            None
        }
    }

    /// Finds the uncle of the current node.
    pub fn uncle(&self) -> Option<&RBTreeNode<K, V>> {
        self.parent()?.sibling()
    }
}

/// A red-black tree balanced binary tree.
#[derive(Debug)]
pub struct RBTree<K: Ord + Clone + PartialEq, V: PartialEq> {
    root_node: Option<Box<RBTreeNode<K, V>>>, // Root node
    num_nodes: usize, // Number of nodes in the tree
}

impl<K: Ord + Clone + PartialEq, V: PartialEq> RBTree<K, V> {
    /// Creates a new, empty red-black tree.
    pub fn new() -> Self {
        RBTree {
            root_node: None,
            num_nodes: 0,
        }
    }

    /// Retrieve the number of entries in the tree.
    pub fn num_entries(&self) -> usize {
        self.num_nodes
    }

    /// Convert the keys in a red-black tree into a Rust Vec. This allows
    /// the tree to be used as an ordered set.
    ///
    /// # Returns
    /// A newly allocated Vec containing all the keys in the tree, in order.
    /// The length of the Vec is equal to the number of entries in the tree.
    pub fn to_vec(&self) -> Vec<K> {
        let mut result = Vec::new();
        self.traverse_in_order(&mut result, self.root_node.as_deref());
        result
    }

    /// Helper function to traverse the tree in-order and collect keys.
    fn traverse_in_order(&self, result: &mut Vec<K>, node: Option<&RBTreeNode<K, V>>) {
        if let Some(node) = node {
            self.traverse_in_order(result, node.child(RBTreeNodeSide::Left));
            result.push(node.key.clone()); // Clone the key to avoid moving it
            self.traverse_in_order(result, node.child(RBTreeNodeSide::Right));
        }
    }

    /// Search a red-black tree for a node with a particular key.
    /// This uses the tree as a mapping.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to search for.
    ///
    /// # Returns
    ///
    /// The tree node containing the given key, or `None` if no entry with the given key is found.
    pub fn lookup_node(&self, key: &K) -> Option<&RBTreeNode<K, V>> {
        let mut node = self.root_node.as_deref();

        while let Some(current_node) = node {
            match key.cmp(&current_node.key) {
                std::cmp::Ordering::Equal => return Some(current_node),
                std::cmp::Ordering::Less => node = current_node.children[0].as_deref(),
                std::cmp::Ordering::Greater => node = current_node.children[1].as_deref(),
            }
        }

        None
    }

    /// Find the root node of a tree.
    ///
    /// # Returns
    /// The root node of the tree, or `None` if the tree is empty.
    pub fn root_node(&self) -> Option<&RBTreeNode<K, V>> {
        self.root_node.as_deref()
    }

    /// Remove a node from the red-black tree.
    ///
    /// # Arguments
    ///
    /// * `node` - The node to remove.
    pub fn remove_node(&mut self, node: &mut RBTreeNode<K, V>) -> Result<(), &'static str> {
        // TODO: Implement the removal logic for the red-black tree.
        // This will involve rebalancing the tree after the node is removed.
        unimplemented!();
    }
}

impl<K: Ord + Clone + PartialEq, V: PartialEq> Drop for RBTree<K, V> {
    /// Automatically frees the tree when it goes out of scope.
    fn drop(&mut self) {
        if let Some(root_node) = self.root_node.take() {
            RBTreeNode::free_subtree(Some(root_node));
        }
    }
}

impl<K: Ord + Clone + PartialEq, V: PartialEq> RBTree<K, V> {
    /// Search a red-black tree for a value corresponding to a particular key.
    /// This uses the tree as a mapping.  Note that this performs
    /// identically to `lookup_node`, except that the value
    /// at the node is returned rather than the node itself.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to search for.
    ///
    /// # Returns
    ///
    /// The value associated with the given key, or `None` if no entry with the given key is found.
    pub fn lookup(&self, key: &K) -> Option<&V> {
        self.lookup_node(key).map(|node| &node.value)
    }
}