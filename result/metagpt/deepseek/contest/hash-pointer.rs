// Rust实现指针哈希函数
// 从C版本转换而来

// 指针哈希函数
pub fn pointer_hash<T>(location: *const T) -> u32 {
    (location as usize) as u32
}

// 为Rust的Hash trait提供适配器
pub struct PointerHasher;

impl std::hash::Hasher for PointerHasher {
    fn finish(&self) -> u64 {
        0
    }
    
    fn write(&mut self, bytes: &[u8]) {
        // 实现可以根据需要调整
    }
}

// 直接返回指针地址作为哈希(保持与C版本相同的行为)
pub fn pointer_hash_simple<T>(location: *const T) -> u32 {
    (location as usize) as u32
}