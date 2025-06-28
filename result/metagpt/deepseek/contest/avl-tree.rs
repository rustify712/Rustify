// Rust implementation of AVL Tree
// Translated from C version

use std::cmp::Ordering;

pub type AVLTreeKey = i32;
pub type AVLTreeValue = i32;

#[derive(Debug)]
pub enum AVLTreeNodeDirection {
    Left,
    Right,
}

#[derive(Debug)]
pub struct AVLTreeNode {
    children: [Option<Box<AVLTreeNode>>; 2],
    parent: Option<*mut AVLTreeNode>,
    key: AVLTreeKey,
    value: AVLTreeValue,
    height: i32,
}

#[derive(Debug)]
pub struct AVLTree {
    root_node: Option<Box<AVLTreeNode>>,
    compare_func: fn(AVLTreeKey, AVLTreeKey) -> Ordering,
    num_nodes: usize,
}

impl AVLTreeNode {
    pub fn new(key: AVLTreeKey, value: AVLTreeValue) -> Self {
        AVLTreeNode {
            children: [None, None],
            parent: None,
            key,
            value,
            height: 1,
        }
    }

    pub fn update_height(&mut self) {
        let left_height = self.children[0].as_ref().map_or(0, |n| n.height);
        let right_height = self.children[1].as_ref().map_or(0, |n| n.height);
        self.height = 1 + std::cmp::max(left_height, right_height);
    }

    pub fn balance_factor(&self) -> i32 {
        let left_height = self.children[0].as_ref().map_or(0, |n| n.height);
        let right_height = self.children[1].as_ref().map_or(0, |n| n.height);
        left_height - right_height
    }
}

impl AVLTree {
    pub fn new(compare_func: fn(AVLTreeKey, AVLTreeKey) -> Ordering) -> Option<Self> {
        Some(AVLTree {
            root_node: None,
            compare_func,
            num_nodes: 0,
        })
    }
}