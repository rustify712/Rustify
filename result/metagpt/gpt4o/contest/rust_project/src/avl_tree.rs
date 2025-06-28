// avl_tree.rs

/// AVL Tree implementation in Rust.
///
/// This implementation provides similar functionality to the C version,
/// with automatic balancing and various utility functions.

use std::cmp::Ordering;

#[derive(Debug)]
struct AVLTreeNode<K, V> {
    key: K,
    value: V,
    height: i32,
    left: Option<Box<AVLTreeNode<K, V>>>,
    right: Option<Box<AVLTreeNode<K, V>>>,
}

impl<K, V> AVLTreeNode<K, V> {
    fn new(key: K, value: V) -> Self {
        AVLTreeNode {
            key,
            value,
            height: 1,
            left: None,
            right: None,
        }
    }
}

pub struct AVLTree<K, V> {
    root: Option<Box<AVLTreeNode<K, V>>>,
    compare_func: fn(&K, &K) -> Ordering,
}

impl<K: Ord, V> AVLTree<K, V> {
    pub fn new(compare_func: fn(&K, &K) -> Ordering) -> Self {
        AVLTree {
            root: None,
            compare_func,
        }
    }

    fn height(node: &Option<Box<AVLTreeNode<K, V>>>) -> i32 {
        node.as_ref().map_or(0, |n| n.height)
    }

    fn update_height(node: &mut Box<AVLTreeNode<K, V>>) {
        let left_height = Self::height(&node.left);
        let right_height = Self::height(&node.right);
        node.height = 1 + std::cmp::max(left_height, right_height);
    }

    fn balance_factor(node: &Option<Box<AVLTreeNode<K, V>>>) -> i32 {
        if let Some(n) = node {
            Self::height(&n.left) - Self::height(&n.right)
        } else {
            0
        }
    }

    fn rotate_left(mut node: Box<AVLTreeNode<K, V>>) -> Box<AVLTreeNode<K, V>> {
        let mut new_root = node.right.take().expect("Right child must exist for left rotation");
        node.right = new_root.left.take();
        Self::update_height(&mut node);
        new_root.left = Some(node);
        Self::update_height(&mut new_root);
        new_root
    }

    fn rotate_right(mut node: Box<AVLTreeNode<K, V>>) -> Box<AVLTreeNode<K, V>> {
        let mut new_root = node.left.take().expect("Left child must exist for right rotation");
        node.left = new_root.right.take();
        Self::update_height(&mut node);
        new_root.right = Some(node);
        Self::update_height(&mut new_root);
        new_root
    }

    fn balance(mut node: Box<AVLTreeNode<K, V>>) -> Box<AVLTreeNode<K, V>> {
        Self::update_height(&mut node);
        let balance = Self::balance_factor(&Some(node.clone()));

        if balance > 1 {
            if Self::balance_factor(&node.left) < 0 {
                node.left = Some(Self::rotate_left(node.left.take().unwrap()));
            }
            return Self::rotate_right(node);
        }

        if balance < -1 {
            if Self::balance_factor(&node.right) > 0 {
                node.right = Some(Self::rotate_right(node.right.take().unwrap()));
            }
            return Self::rotate_left(node);
        }

        node
    }

    pub fn insert(&mut self, key: K, value: V) {
        self.root = Some(Self::insert_node(self.root.take(), key, value, self.compare_func));
    }

    fn insert_node(
        node: Option<Box<AVLTreeNode<K, V>>>,
        key: K,
        value: V,
        compare_func: fn(&K, &K) -> Ordering,
    ) -> Box<AVLTreeNode<K, V>> {
        if let Some(mut n) = node {
            match compare_func(&key, &n.key) {
                Ordering::Less => n.left = Some(Self::insert_node(n.left.take(), key, value, compare_func)),
                Ordering::Greater => n.right = Some(Self::insert_node(n.right.take(), key, value, compare_func)),
                Ordering::Equal => n.value = value,
            }
            Self::balance(n)
        } else {
            Box::new(AVLTreeNode::new(key, value))
        }
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        Self::get_node(&self.root, key, self.compare_func)
    }

    fn get_node<'a>(
        node: &'a Option<Box<AVLTreeNode<K, V>>>,
        key: &K,
        compare_func: fn(&K, &K) -> Ordering,
    ) -> Option<&'a V> {
        if let Some(n) = node {
            match compare_func(key, &n.key) {
                Ordering::Less => Self::get_node(&n.left, key, compare_func),
                Ordering::Greater => Self::get_node(&n.right, key, compare_func),
                Ordering::Equal => Some(&n.value),
            }
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compare_ints(a: &i32, b: &i32) -> Ordering {
        a.cmp(b)
    }

    #[test]
    fn test_avl_tree_insert_and_get() {
        let mut tree = AVLTree::new(compare_ints);
        tree.insert(10, "value10");
        tree.insert(20, "value20");
        tree.insert(5, "value5");

        assert_eq!(tree.get(&10), Some(&"value10"));
        assert_eq!(tree.get(&20), Some(&"value20"));
        assert_eq!(tree.get(&5), Some(&"value5"));
        assert_eq!(tree.get(&15), None);
    }
}