use std::cell::Cell;

/// 表示 Trie 中的一个节点。
pub enum TrieNodeData<T> {
    Value(Option<T>),
    FreeList(Option<Box<TrieNode<T>>>),
}

pub struct TrieNode<T> {
    /// 存储在节点中的值。
    pub data: TrieNodeData<T>,
    /// 记录该节点被使用的次数。
    pub use_count: Cell<u32>,
    /// 指向下一层节点的指针数组。
    pub next: Vec<Option<Box<TrieNode<T>>>>,
}

impl<T> TrieNode<T> {
    /// 创建一个新的 `TrieNode`。
    pub fn new() -> Self {
        TrieNode {
            data: TrieNodeData::Value(None),
            use_count: Cell::new(0),
            next: Vec::with_capacity(256), // 使用 Vec::with_capacity 初始化
        }
    }

    /// 将一个节点插入到链表的头部。
    ///
    /// # 参数
    /// - `list`: 链表的头节点指针的可变引用。
    /// - `node`: 要插入的节点。
    pub fn free_list_push(list: &mut Option<Box<TrieNode<T>>>, mut node: Box<TrieNode<T>>) {
        node.data = TrieNodeData::FreeList(list.take());
        *list = Some(node);
    }

    /// 从自由列表中弹出一个节点。
    pub fn free_list_pop(list: &mut Option<Box<TrieNode<T>>>) -> Option<Box<TrieNode<T>>> {
        list.take().map(|mut node| {
            if let TrieNodeData::FreeList(ref mut data) = node.data { // 使用 ref mut 借用
                *list = data.take();
            }
            node
        })
    }
}

/// A trie structure.
pub struct Trie<T> {
    /// The root node of the trie.
    pub root_node: Option<Box<TrieNode<T>>>,
}

impl<T: Clone> Trie<T> {
    /// 销毁一个 Trie 数据结构，释放所有节点和 Trie 本身的内存。
    pub fn free(&mut self) {
        let mut free_list = None;

        // 将根节点添加到自由列表中
        if let Some(root_node) = self.root_node.take() {
            TrieNode::free_list_push(&mut free_list, root_node);
        }

        // 遍历自由列表，释放每个节点的内存，并将该节点的所有子节点添加到自由列表中
        while let Some(mut node) = TrieNode::free_list_pop(&mut free_list) {
            for i in 0..256 {
                if let Some(child) = node.next[i].take() {
                    TrieNode::free_list_push(&mut free_list, child);
                }
            }
            // 节点内存自动释放，因为 Box 会在作用域结束时自动释放
        }

        // Trie 本身的内存自动释放，因为 Trie 是栈上的变量
    }

    /// 查找二进制键对应的节点。
    ///
    /// # Arguments
    ///
    /// * `key` - The binary key to search for.
    ///
    /// # Returns
    ///
    /// An `Option<&TrieNode<T>>` representing the end node if found, or `None` if not found.
    pub fn find_end_binary(&self, key: &[u8]) -> Option<&TrieNode<T>> {
        let mut node = self.root_node.as_ref()?;

        for &c in key {
            node = node.next[c as usize].as_ref()?;
        }

        Some(node)
    }

    /// Look up a value from its key in a trie.
    /// The key is a sequence of bytes; for a key that is a NUL-terminated
    /// text string, use the `lookup` method.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to look up.
    ///
    /// # Returns
    ///
    /// The value associated with the key, or `None` if not found in the trie.
    pub fn lookup_binary(&self, key: &[u8]) -> Option<T> {
        self.find_end_binary(key).and_then(|node| {
            if let TrieNodeData::Value(value) = &node.data {
                value.clone() // 直接返回 value，而不是克隆
            } else {
                None
            }
        })
    }

    /// Insert a new key-value pair into the trie. The key is a sequence of bytes.
    /// For a key that is a NUL-terminated text string, use `insert`.
    ///
    /// # Arguments
    ///
    /// * `key` - The key to access the new value.
    /// * `value` - The value to insert.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the value was inserted successfully, or `Err(())` if it was not possible to allocate memory for the new entry.
    pub fn insert_binary(&mut self, key: &[u8], value: Option<T>) -> Result<(), ()> {
        // Cannot insert NULL values
        if value.is_none() {
            return Err(());
        }

        // Search to see if this is already in the tree
        let mut rover = &mut self.root_node;
        let mut node = rover.as_mut();
        let mut p = 0;

        loop {
            if node.is_none() {
                // Node does not exist, so create it
                let new_node = Box::new(TrieNode::new());
                *rover = Some(new_node);
                node = rover.as_mut();
            }

            // Increase the node use count
            let use_count = node.as_ref().unwrap().use_count.get();
            node.as_mut().unwrap().use_count.set(use_count + 1);

            // Current character
            let c = key[p] as usize;

            // Reached the end of string? If so, we're finished.
            if p == key.len() {
                node.as_mut().unwrap().data = TrieNodeData::Value(value);
                break;
            }

            // Advance to the next node in the chain
            rover = &mut rover.as_mut().unwrap().next[c];
            node = rover.as_mut();
            p += 1;
        }

        Ok(())
    }

    /// 回滚在插入过程中创建的节点，释放这些节点的内存，并更新相关的指针。
    pub fn insert_rollback(&mut self, key: &[u8]) {
        let mut node = self.root_node.take();
        let mut prev_ptr = &mut self.root_node;
        let mut p = 0;

        while let Some(mut current_node) = node {
            // 找到下一个节点
            let next_prev_ptr = &mut current_node.next[key[p] as usize];
            let next_node = next_prev_ptr.take();
            p += 1;

            // 减少 use_count 并释放节点
            current_node.use_count.set(current_node.use_count.get() - 1);
            if current_node.use_count.get() == 0 {
                // 释放节点
                drop(current_node);
                *prev_ptr = None;
                *next_prev_ptr = None;
            } else {
                *prev_ptr = Some(current_node);
            }

            // 更新指针
            node = next_node;
            prev_ptr = next_prev_ptr;
        }
    }

    /// Remove an entry from the trie.
    /// The key is a NUL-terminated string; for binary strings, use `remove_binary`.
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the entry to remove.
    ///
    /// # Returns
    ///
    /// `true` if the key was removed successfully, or `false` if it is not present in the trie.
    pub fn remove(&mut self, key: &str) -> bool {
        // Find the end node and remove the value
        let end_node = self.find_end(key);
        if let Some(node) = end_node {
            if let TrieNodeData::Value(Some(_)) = node.data {
                node.data = TrieNodeData::Value(None);
            } else {
                return false;
            }
        } else {
            return false;
        }

        // Now traverse the tree again as before, decrementing the use count of each node.
        // Free back nodes as necessary.
        let mut node = self.root_node.as_mut();
        let mut last_next_ptr = &mut self.root_node;
        let mut chars = key.chars();

        while let Some(c) = chars.next() {
            if let Some(current_node) = node {
                let next_index = c as usize;
                let next_node = current_node.next[next_index].take();
                if let Some(next) = next_node {
                    // Decrease the use count and free the node if it reaches zero.
                    current_node.use_count.set(current_node.use_count.get() - 1);
                    if current_node.use_count.get() == 0 {
                        // Free the node
                        *last_next_ptr = None;
                    }
                    last_next_ptr = &mut current_node.next[next_index];
                    node = Some(next);
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        true
    }

    /// Remove an entry from a trie.
    /// The key is a sequence of bytes; for a key that is a NUL-terminated
    /// text string, use the `remove` method.
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the entry to remove.
    ///
    /// # Returns
    ///
    /// `true` if the key was removed successfully, or `false` if it is not present in the trie.
    pub fn remove_binary(&mut self, key: &[u8]) -> bool {
        // Find the end node and remove the value
        let mut node = self.find_end_binary(key)?;

        if let TrieNodeData::Value(ref mut value) = node.data {
            if value.is_some() {
                *value = None;
            } else {
                return false;
            }
        } else {
            return false;
        }

        // Now traverse the tree again as before, decrementing the use
        // count of each node. Free back nodes as necessary.
        let mut node = self.root_node.as_mut()?;
        let mut last_next_ptr = &mut self.root_node;
        let mut p = 0;

        loop {
            // Find the next node
            let c = key[p] as usize;
            let next = node.next[c].take();

            // Free this node if necessary
            node.use_count.set(node.use_count.get() - 1);

            if node.use_count.get() == 0 {
                // Set the "next" pointer on the previous node to None,
                // to unlink the freed node from the tree.
                *last_next_ptr = None;
            }

            // Go to the next character or finish
            if p == key.len() {
                break;
            } else {
                p += 1;
            }

            // If necessary, save the location of the "next" pointer
            // so that it may be set to None on the next iteration if
            // the next node visited is freed.
            last_next_ptr = &mut node.next[c];

            // Jump to the next node
            node = next.as_mut()?;
        }

        // Removed successfully
        true
    }
}