// Rust实现集合
// 从C版本转换而来

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub type SetValue = i32;
pub type SetHashFunc = fn(&SetValue) -> u64;
pub type SetEqualFunc = fn(&SetValue, &SetValue) -> bool;
pub type SetFreeFunc = fn(SetValue);

struct SetEntry {
    data: SetValue,
    next: Option<Box<SetEntry>>,
}

pub struct Set {
    table: Vec<Option<Box<SetEntry>>>,
    entries: usize,
    table_size: usize,
    prime_index: usize,
    hash_func: SetHashFunc,
    equal_func: SetEqualFunc,
    free_func: Option<SetFreeFunc>,
}

// 哈希表质数表
const SET_PRIMES: [usize; 24] = [
    193, 389, 769, 1543, 3079, 6151, 12289, 24593, 49157, 98317,
    196613, 393241, 786433, 1572869, 3145739, 6291469,
    12582917, 25165843, 50331653, 100663319, 201326611,
    402653189, 805306457, 1610612741,
];

impl Set {
    pub fn new(hash_func: SetHashFunc, equal_func: SetEqualFunc) -> Self {
        Set {
            table: Vec::new(),
            entries: 0,
            table_size: 0,
            prime_index: 0,
            hash_func,
            equal_func,
            free_func: None,
        }
    }
}