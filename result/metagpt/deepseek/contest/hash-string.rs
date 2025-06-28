// Rust实现字符串哈希函数
// 从C版本转换而来

// djb2字符串哈希函数
pub fn string_hash(s: &str) -> u32 {
    let mut hash: u32 = 5381;
    
    for c in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(c as u32);
    }
    
    hash
}

// 大小写不敏感的djb2字符串哈希函数
pub fn string_nocase_hash(s: &str) -> u32 {
    let mut hash: u32 = 5381;
    
    for c in s.bytes() {
        hash = hash.wrapping_mul(33).wrapping_add(c.to_ascii_lowercase() as u32);
    }
    
    hash
}

// 为Rust的Hash trait提供适配器
pub struct StringHasher {
    hash: u32,
    nocase: bool,
}

impl StringHasher {
    pub fn new(nocase: bool) -> Self {
        StringHasher {
            hash: 5381,
            nocase,
        }
    }
}

impl std::hash::Hasher for StringHasher {
    fn finish(&self) -> u64 {
        self.hash as u64
    }
    
    fn write(&mut self, bytes: &[u8]) {
        for &c in bytes {
            let c = if self.nocase { c.to_ascii_lowercase() } else { c };
            self.hash = self.hash.wrapping_mul(33).wrapping_add(c as u32);
        }
    }
}