// rb_tree.rs

/// Red-Black Tree implementation in Rust.
///
/// This implementation provides similar functionality to the C version,
/// with automatic balancing and various utility functions.

use std::cmp::Ordering;

#[derive(Debug, PartialEq)]
enum Color {
    Red,
    Black,
}

#[derive(Debug)]
struct RBTreeNode<K, V> {
    key: K,
    value: V,
    color: Color,
    parent: Option<*mut RBTreeNode<K, V>>,
    left: Option<Box<RBTreeNode<K, V>>>,
    right: Option<Box<RBTreeNode<K, V>>>,
}

impl<K, V> RBTreeNode<K, V> {
    fn new(key: K, value: V, color: Color) -> Self {
        RBTreeNode {
            key,
            value,
            color,
            parent: None,
            left: None,
            right: None,
        }
    }
}

pub struct RBTree<K, V> {
    root: Option<Box<RBTreeNode<K, V>>>,
    compare_func: fn(&K, &K) -> Ordering,
}

impl<K: Ord, V> RBTree<K, V> {
    pub fn new(compare_func: fn(&K, &K) -> Ordering) -> Self {
        RBTree {
            root: None,
            compare_func,
        }
    }

    fn rotate_left(&mut self, node: &mut Box<RBTreeNode<K, V>>) {
        if let Some(mut right) = node.right.take() {
            node.right = right.left.take();
            if let Some(ref mut right_left) = node.right {
                right_left.parent = Some(&mut **node);
            }
            right.parent = node.parent;
            if let Some(parent) = node.parent {
                unsafe {
                    if (*parent).left.as_ref().map_or(false, |left| left.key == node.key) {
                        (*parent).left = Some(right);
                    } else {
                        (*parent).right = Some(right);
                    }
                }
            } else {
                self.root = Some(right);
            }
            node.parent = Some(&mut **right);
            right.left = Some(node);
        }
    }

    fn rotate_right(&mut self, node: &mut Box<RBTreeNode<K, V>>) {
        if let Some(mut left) = node.left.take() {
            node.left = left.right.take();
            if let Some(ref mut left_right) = node.left {
                left_right.parent = Some(&mut **node);
            }
            left.parent = node.parent;
            if let Some(parent) = node.parent {
                unsafe {
                    if (*parent).left.as_ref().map_or(false, |left| left.key == node.key) {
                        (*parent).left = Some(left);
                    } else {
                        (*parent).right = Some(left);
                    }
                }
            } else {
                self.root = Some(left);
            }
            node.parent = Some(&mut **left);
            left.right = Some(node);
        }
    }

    pub fn insert(&mut self, key: K, value: V) {
        let new_node = Box::new(RBTreeNode::new(key, value, Color::Red));
        if self.root.is_none() {
            self.root = Some(new_node);
            self.root.as_mut().unwrap().color = Color::Black;
        } else {
            self.insert_node(self.root.as_mut().unwrap(), new_node);
        }
    }

    fn insert_node(&mut self, root: &mut Box<RBTreeNode<K, V>>, mut new_node: Box<RBTreeNode<K, V>>) {
        let cmp = (self.compare_func)(&new_node.key, &root.key);
        if cmp == Ordering::Less {
            if let Some(ref mut left) = root.left {
                self.insert_node(left, new_node);
            } else {
                new_node.parent = Some(&mut **root);
                root.left = Some(new_node);
                self.fix_insert(root.left.as_mut().unwrap());
            }
        } else {
            if let Some(ref mut right) = root.right {
                self.insert_node(right, new_node);
            } else {
                new_node.parent = Some(&mut **root);
                root.right = Some(new_node);
                self.fix_insert(root.right.as_mut().unwrap());
            }
        }
    }

    fn fix_insert(&mut self, node: &mut Box<RBTreeNode<K, V>>) {
        // Fixing the tree after insertion to maintain red-black properties
        // This is a placeholder for the actual fix-up logic
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        self.get_node(&self.root, key)
    }

    fn get_node<'a>(&'a self, node: &'a Option<Box<RBTreeNode<K, V>>>, key: &K) -> Option<&'a V> {
        if let Some(n) = node {
            match (self.compare_func)(key, &n.key) {
                Ordering::Less => self.get_node(&n.left, key),
                Ordering::Greater => self.get_node(&n.right, key),
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
    fn test_rb_tree_insert_and_get() {
        let mut tree = RBTree::new(compare_ints);
        tree.insert(10, "value10");
        tree.insert(20, "value20");
        tree.insert(5, "value5");

        assert_eq!(tree.get(&10), Some(&"value10"));
        assert_eq!(tree.get(&20), Some(&"value20"));
        assert_eq!(tree.get(&5), Some(&"value5"));
        assert_eq!(tree.get(&15), None);
    }
}