// Rust实现哈希表
// 从C版本转换而来

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub type HashTableHashFunc<K> = fn(&K) -> u64;
pub type HashTableEqualFunc<K> = fn(&K, &K) -> bool;
pub type HashTableKeyFreeFunc<K> = fn(K);
pub type HashTableValueFreeFunc<V> = fn(V);

pub struct HashTablePair<K, V> {
    pub key: K,
    pub value: V,
}

struct HashTableEntry<K, V> {
    pair: HashTablePair<K, V>,
    next: Option<Box<HashTableEntry<K, V>>>,
}

pub struct HashTable<K, V> {
    table: Vec<Option<Box<HashTableEntry<K, V>>>>,
    table_size: usize,
    hash_func: HashTableHashFunc<K>,
    equal_func: HashTableEqualFunc<K>,
    key_free_func: Option<HashTableKeyFreeFunc<K>>,
    value_free_func: Option<HashTableValueFreeFunc<V>>,
    entries: usize,
}

// 哈希表质数表
const HASH_TABLE_PRIMES: [usize; 24] = [
    193, 389, 769, 1543, 3079, 6151, 12289, 24593, 49157, 98317,
    196613, 393241, 786433, 1572869, 3145739, 6291469,
    12582917, 25165843, 50331653, 100663319, 201326611,
    402653189, 805306457, 1610612741,
];

impl<K, V> HashTable<K, V> {
    pub fn new(
        hash_func: HashTableHashFunc<K>,
        equal_func: HashTableEqualFunc<K>,
        key_free_func: Option<HashTableKeyFreeFunc<K>>,
        value_free_func: Option<HashTableValueFreeFunc<V>>,
    ) -> Self {
        let table_size = HASH_TABLE_PRIMES[0];
        
        HashTable {
            table: vec![None; table_size],
            table_size,
            hash_func,
            equal_func,
            key_free_func,
            value_free_func,
            entries: 0,
        }
    }

    // 计算键的哈希值并确定桶位置
    fn calculate_bucket(&self, key: &K) -> usize {
        let hash = (self.hash_func)(key);
        (hash % self.table_size as u64) as usize
    }

    // 插入键值对
    pub fn insert(&mut self, key: K, value: V) -> bool {
        let bucket = self.calculate_bucket(&key);
        
        // 检查键是否已存在
        let mut current = &mut self.table[bucket];
        while let Some(entry) = current {
            if (self.equal_func)(&entry.pair.key, &key) {
                // 替换现有值
                if let Some(free_fn) = self.value_free_func {
                    free_fn(entry.pair.value);
                }
                entry.pair.value = value;
                return true;
            }
            current = &mut entry.next;
        }
        
        // 创建新条目
        let new_entry = Box::new(HashTableEntry {
            pair: HashTablePair { key, value },
            next: None,
        });
        
        *current = Some(new_entry);
        self.entries += 1;
        
        true
    }

    // 查找键对应的值
    pub fn lookup(&self, key: &K) -> Option<&V> {
        let bucket = self.calculate_bucket(key);
        
        let mut current = &self.table[bucket];
        while let Some(entry) = current {
            if (self.equal_func)(&entry.pair.key, key) {
                return Some(&entry.pair.value);
            }
            current = &entry.next;
        }
        
        None
    }

    // 删除键值对
    pub fn remove(&mut self, key: &K) -> bool {
        let bucket = self.calculate_bucket(key);
        
        let mut prev = None;
        let mut current = &mut self.table[bucket];
        
        while let Some(entry) = current {
            if (self.equal_func)(&entry.pair.key, key) {
                // 释放键和值
                if let Some(free_fn) = self.key_free_func {
                    free_fn(entry.pair.key);
                }
                if let Some(free_fn) = self.value_free_func {
                    free_fn(entry.pair.value);
                }
                
                // 从链表中移除
                *current = entry.next.take();
                self.entries -= 1;
                return true;
            }
            prev = current;
            current = &mut entry.next;
        }
        
        false
    }
}