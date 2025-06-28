// trie.rs

/// Trie implementation in Rust.
///
/// This implementation provides similar functionality to the C version,
/// with operations to manipulate the trie structure.

use std::collections::HashMap;

pub struct TrieNode<T> {
    data: Option<T>,
    children: HashMap<u8, TrieNode<T>>,
}

impl<T> TrieNode<T> {
    fn new() -> Self {
        TrieNode {
            data: None,
            children: HashMap::new(),
        }
    }
}

pub struct Trie<T> {
    root: TrieNode<T>,
}

impl<T> Trie<T> {
    /// Create a new empty trie.
    pub fn new() -> Self {
        Trie {
            root: TrieNode::new(),
        }
    }

    /// Insert a value into the trie with a string key.
    ///
    /// # Arguments
    /// * `key` - The key to insert.
    /// * `value` - The value to associate with the key.
    pub fn insert(&mut self, key: &str, value: T) {
        let mut node = &mut self.root;
        for &byte in key.as_bytes() {
            node = node.children.entry(byte).or_insert_with(TrieNode::new);
        }
        node.data = Some(value);
    }

    /// Retrieve a value from the trie by a string key.
    ///
    /// # Arguments
    /// * `key` - The key to look up.
    ///
    /// # Returns
    /// A reference to the value, or `None` if not found.
    pub fn get(&self, key: &str) -> Option<&T> {
        let mut node = &self.root;
        for &byte in key.as_bytes() {
            match node.children.get(&byte) {
                Some(next_node) => node = next_node,
                None => return None,
            }
        }
        node.data.as_ref()
    }

    /// Remove a value from the trie by a string key.
    ///
    /// # Arguments
    /// * `key` - The key to remove.
    ///
    /// # Returns
    /// `true` if the key was removed, `false` if it was not found.
    pub fn remove(&mut self, key: &str) -> bool {
        self.remove_recursive(&mut self.root, key.as_bytes(), 0)
    }

    fn remove_recursive(&mut self, node: &mut TrieNode<T>, key: &[u8], depth: usize) -> bool {
        if depth == key.len() {
            if node.data.is_some() {
                node.data = None;
                return node.children.is_empty();
            }
            return false;
        }

        let byte = key[depth];
        if let Some(child_node) = node.children.get_mut(&byte) {
            if self.remove_recursive(child_node, key, depth + 1) {
                node.children.remove(&byte);
                return node.data.is_none() && node.children.is_empty();
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trie_operations() {
        let mut trie = Trie::new();
        trie.insert("hello", 1);
        trie.insert("world", 2);

        assert_eq!(trie.get("hello"), Some(&1));
        assert_eq!(trie.get("world"), Some(&2));
        assert_eq!(trie.get("hell"), None);

        assert!(trie.remove("hello"));
        assert_eq!(trie.get("hello"), None);
        assert_eq!(trie.get("world"), Some(&2));
    }
}