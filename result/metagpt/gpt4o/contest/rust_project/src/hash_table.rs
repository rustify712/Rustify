// hash_table.rs

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::ptr;

/// A pair of key and value stored in the hash table.
pub struct HashTablePair<K, V> {
    pub key: K,
    pub value: V,
}

/// An entry in the hash table.
struct HashTableEntry<K, V> {
    pair: HashTablePair<K, V>,
    next: Option<Box<HashTableEntry<K, V>>>,
}

/// The hash table structure.
pub struct HashTable<K, V> {
    table: Vec<Option<Box<HashTableEntry<K, V>>>>,
    table_size: usize,
    entries: usize,
    prime_index: usize,
}

impl<K: Eq + Hash, V> HashTable<K, V> {
    /// Create a new hash table.
    pub fn new() -> Self {
        let initial_size = 193; // Starting with the first prime number
        HashTable {
            table: vec![None; initial_size],
            table_size: initial_size,
            entries: 0,
            prime_index: 0,
        }
    }

    /// Hash function for the key.
    fn hash(&self, key: &K) -> usize {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        (hasher.finish() % self.table_size as u64) as usize
    }

    /// Insert a key-value pair into the hash table.
    pub fn insert(&mut self, key: K, value: V) {
        let index = self.hash(&key);
        let new_entry = Box::new(HashTableEntry {
            pair: HashTablePair { key, value },
            next: self.table[index].take(),
        });
        self.table[index] = Some(new_entry);
        self.entries += 1;

        if self.entries > self.table_size {
            self.enlarge();
        }
    }

    /// Retrieve a value by key from the hash table.
    pub fn get(&self, key: &K) -> Option<&V> {
        let index = self.hash(key);
        let mut entry = &self.table[index];

        while let Some(ref e) = entry {
            if e.pair.key == *key {
                return Some(&e.pair.value);
            }
            entry = &e.next;
        }
        None
    }

    /// Remove a key-value pair from the hash table.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let index = self.hash(key);
        let mut entry = &mut self.table[index];

        while let Some(ref mut e) = entry {
            if e.pair.key == *key {
                let removed_entry = entry.take();
                *entry = removed_entry.unwrap().next.take();
                self.entries -= 1;
                return Some(removed_entry.unwrap().pair.value);
            }
            entry = &mut e.next;
        }
        None
    }

    /// Enlarge the hash table when the load factor is too high.
    fn enlarge(&mut self) {
        let new_size = self.table_size * 2;
        let mut new_table = vec![None; new_size];

        for entry in self.table.drain(..) {
            let mut current_entry = entry;
            while let Some(mut e) = current_entry {
                let index = self.hash(&e.pair.key) % new_size;
                current_entry = e.next.take();
                e.next = new_table[index].take();
                new_table[index] = Some(e);
            }
        }

        self.table = new_table;
        self.table_size = new_size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_get() {
        let mut table = HashTable::new();
        table.insert("key1", "value1");
        table.insert("key2", "value2");

        assert_eq!(table.get(&"key1"), Some(&"value1"));
        assert_eq!(table.get(&"key2"), Some(&"value2"));
        assert_eq!(table.get(&"key3"), None);
    }

    #[test]
    fn test_remove() {
        let mut table = HashTable::new();
        table.insert("key1", "value1");
        table.insert("key2", "value2");

        assert_eq!(table.remove(&"key1"), Some("value1"));
        assert_eq!(table.get(&"key1"), None);
        assert_eq!(table.get(&"key2"), Some(&"value2"));
    }
}