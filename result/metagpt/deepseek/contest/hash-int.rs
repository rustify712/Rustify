// Rust实现整数哈希函数
// 从C版本转换而来

use std::hash::{Hash, Hasher};

// 整数哈希函数
pub fn int_hash<T: Hash>(value: &T) -> u32 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish() as u32
}

// 为Rust的Hash trait提供适配器
pub struct IntHasher;

impl std::hash::Hasher for IntHasher {
    fn finish(&self) -> u64 {
        0
    }
    
    fn write(&mut self, bytes: &[u8]) {
        // 实现可以根据需要调整
    }
}

// 直接返回整数值作为哈希(保持与C版本相同的行为)
pub fn int_hash_simple(value: &i32) -> u32 {
    *value as u32
}