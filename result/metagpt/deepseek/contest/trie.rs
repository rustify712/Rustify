use std::ptr;

type TrieValue = *mut std::ffi::c_void;

struct TrieNode {
    data: TrieValue,
    use_count: u32,
    next: [Option<Box<TrieNode>>; 256],
}

impl TrieNode {
    fn new() -> Self {
        TrieNode {
            data: ptr::null_mut(),
            use_count: 0,
            next: [(); 256].map(|_| None),
        }
    }
}

pub struct Trie {
    root_node: Option<Box<TrieNode>>,
}

impl Trie {
    pub fn new() -> Self {
        Trie {
            root_node: None,
        }
    }

    fn free_list_push(list: &mut Option<Box<TrieNode>>, node: Box<TrieNode>) {
        let mut node = node;
        node.data = list.take().map(Box::into_raw).map(|p| p as _).unwrap_or(ptr::null_mut());
        *list = Some(node);
    }

    fn free_list_pop(list: &mut Option<Box<TrieNode>>) -> Option<Box<TrieNode>> {
        list.take().map(|mut node| {
            let data = node.data;
            if !data.is_null() {
                node.data = ptr::null_mut();
                unsafe { Box::from_raw(data as *mut TrieNode) }
            } else {
                node
            }
        })
    }

    pub fn free(&mut self) {
        let mut free_list = None;

        if let Some(root) = self.root_node.take() {
            Self::free_list_push(&mut free_list, root);
        }

        while let Some(mut node) = Self::free_list_pop(&mut free_list) {
            for i in 0..256 {
                if let Some(child) = node.next[i].take() {
                    Self::free_list_push(&mut free_list, child);
                }
            }
        }
    }

    fn find_end(&self, key: &str) -> Option<&TrieNode> {
        let mut node = self.root_node.as_deref();

        for c in key.bytes() {
            node = node.and_then(|n| n.next[c as usize].as_deref());
        }

        node
    }

    fn find_end_binary(&self, key: &[u8]) -> Option<&TrieNode> {
        let mut node = self.root_node.as_deref();

        for &c in key {
            node = node.and_then(|n| n.next[c as usize].as_deref());
        }

        node
    }

    pub fn insert(&mut self, key: &str, value: TrieValue) -> bool {
        if value.is_null() {
            return false;
        }

        if let Some(node) = self.find_end(key) {
            if !node.data.is_null() {
                unsafe {
                    (node as *const TrieNode as *mut TrieNode).as_mut().unwrap().data = value;
                }
                return true;
            }
        }

        let mut rover = &mut self.root_node;
        let key_bytes = key.as_bytes();
        let mut p = 0;

        loop {
            let c = if p < key_bytes.len() { 
                key_bytes[p] 
            } else { 
                break true 
            };

            let node = rover.take();
            let node = match node {
                Some(n) => n,
                None => {
                    let new_node = Box::new(TrieNode::new());
                    *rover = Some(new_node);
                    rover.as_mut().unwrap().clone()
                }
            };

            let mut node = node;
            node.use_count += 1;
            *rover = Some(node);

            if p == key_bytes.len() - 1 {
                rover.as_mut().unwrap().data = value;
                break true;
            }

            rover = &mut rover.as_mut().unwrap().next[c as usize];
            p += 1;
        }
    }

    pub fn lookup(&self, key: &str) -> Option<TrieValue> {
        self.find_end(key).map(|n| n.data)
    }

    pub fn lookup_binary(&self, key: &[u8]) -> Option<TrieValue> {
        self.find_end_binary(key).map(|n| n.data)
    }
}

impl Drop for Trie {
    fn drop(&mut self) {
        self.free();
    }
}