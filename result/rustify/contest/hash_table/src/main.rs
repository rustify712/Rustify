use std::fmt::Debug;
use std::hash::{Hash, Hasher};

/// Internal structure representing an entry in hash table
/// used as `HashTableIterator` next result.
#[derive(Debug, Clone, PartialEq)]
pub struct HashTablePair<K: Clone, V: Clone> {
    pub key: K,
    pub value: V,
}

/// Type of function used to free keys when entries are removed from a hash table.
pub type HashTableKeyFreeFunc<K> = Option<Box<dyn FnOnce(K)>>;

/// Internal structure representing an entry in a hash table.
#[derive(Debug)]
pub struct HashTableEntry<K: Clone, V: Clone> {
    pub pair: HashTablePair<K, V>,
    pub next: Option<Box<HashTableEntry<K, V>>>,
}

/// A hash table structure.
pub struct HashTable<K: Clone, V: Clone> {
    table: Vec<Option<Box<HashTableEntry<K, V>>>>,
    table_size: usize,
    entries: usize,
    prime_index: usize,
    key_free_func: HashTableKeyFreeFunc<K>,
    value_free_func: Option<Box<dyn FnOnce(V)>>,
}

const hash_table_primes: [usize; 10] = [
    53, 97, 193, 389, 769, 1543, 3079, 6151, 12289, 24593,
];

impl<K: Debug + Clone, V: Debug + Clone> Debug for HashTable<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HashTable")
            .field("table", &self.table)
            .field("table_size", &self.table_size)
            .field("entries", &self.entries)
            .field("prime_index", &self.prime_index)
            .finish()
    }
}

impl<K: Clone, V: Clone> HashTable<K, V> {
    /// Retrieve the number of entries in a hash table.
    pub fn num_entries(&self) -> usize {
        self.entries
    }

    /// Allocate a new table for the hash table.
    pub fn allocate_table(&mut self) -> Result<(), &'static str> {
        let new_table_size = if self.prime_index < hash_table_primes.len() {
            hash_table_primes[self.prime_index]
        } else {
            self.entries * 10
        };

        self.table_size = new_table_size;
        self.table = Vec::with_capacity(new_table_size);
        self.table.resize_with(new_table_size, || None);

        Ok(())
    }

    /// Free an entry in the hash table.
    pub fn free_entry(&mut self, entry: Box<HashTableEntry<K, V>>) {
        if let Some(key_free_func) = self.key_free_func.take() {
            key_free_func(entry.pair.key);
        }
        if let Some(value_free_func) = self.value_free_func.take() {
            value_free_func(entry.pair.value);
        }
    }

    /// Register functions used to free the key and value when an entry is removed from a hash table.
    pub fn register_free_functions(&mut self, key_free_func: Option<Box<dyn FnOnce(K)>>, value_free_func: Option<Box<dyn FnOnce(V)>>) {
        self.key_free_func = key_free_func;
        self.value_free_func = value_free_func;
    }

    /// Create a new hash table.
    ///
    /// # Type Parameters
    ///
    /// * `K` - The type of the keys in the hash table.
    /// * `V` - The type of the values in the hash table.
    ///
    /// # Returns
    ///
    /// A new hash table structure, or an error if it was not possible to allocate the new hash table.
    pub fn new() -> Result<HashTable<K, V>, &'static str> {
        let mut hash_table = HashTable {
            table: Vec::new(),
            table_size: 0,
            entries: 0,
            prime_index: 0,
            key_free_func: None,
            value_free_func: None,
        };
        hash_table.allocate_table()?;
        Ok(hash_table)
    }

    /// Destroy a hash table.
    pub fn free(&mut self) {
        // 将 `self.table` 的借用作用域与 `self.free_entry` 的借用作用域分开
        let mut table = std::mem::take(&mut self.table);

        for mut entry in table.iter_mut() {
            while let Some(boxed_entry) = entry.take() {
                self.free_entry(boxed_entry);
            }
        }

        // 将 `table` 恢复到 `self.table`
        self.table = table;
    }
}

impl<K: Hash + Eq + Clone, V: Clone> HashTable<K, V> {
    /// Insert a value into the hash table, overwriting any existing entry with the same key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key for the new value.
    /// * `value` - The value to insert.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the value was added successfully, or `Err("Memory allocation failed")` if it was not possible to allocate memory for the new entry.
    pub fn insert(&mut self, key: K, value: V) -> Result<(), &'static str> {
        // Check if the table needs to be enlarged
        if (self.entries * 3) / self.table_size > 0 {
            if !self.enlarge() {
                return Err("Failed to enlarge the table");
            }
        }

        // Generate the hash of the key and hence the index into the table
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        let index = (hasher.finish() % self.table_size as u64) as usize;

        // Traverse the chain at this location and look for an existing entry with the same key
        let mut rover = self.table[index].as_mut();

        while let Some(entry) = rover {
            if entry.pair.key == key {
                // Same key: overwrite this entry with new data
                entry.pair.value = value;
                return Ok(());
            }
            rover = entry.next.as_mut();
        }

        // Not in the hash table yet. Create a new entry
        let new_entry = Box::new(HashTableEntry {
            pair: HashTablePair { key, value },
            next: self.table[index].take(),
        });

        self.table[index] = Some(new_entry);
        self.entries += 1;

        Ok(())
    }

    /// Enlarge the hash table by rehashing all entries into a new, larger table.
    fn enlarge(&mut self) -> bool {
        let old_table = std::mem::replace(&mut self.table, Vec::new());
        let old_table_size = self.table_size;

        // Allocate a new, larger table
        self.prime_index += 1;
        if !self.allocate_table().is_err() {
            self.table_size = if self.prime_index < hash_table_primes.len() {
                hash_table_primes[self.prime_index]
            } else {
                self.entries * 10
            };

            // Rehash all entries from the old table into the new table
            for mut entry in old_table.into_iter().flatten() {
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                entry.pair.key.hash(&mut hasher);
                let index = (hasher.finish() % self.table_size as u64) as usize;

                entry.next = self.table[index].take();
                self.table[index] = Some(entry);
            }

            true
        } else {
            self.table = old_table;
            self.table_size = old_table_size;
            false
        }
    }

    /// Look up a value in the hash table by key.
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the value to look up.
    ///
    /// # Returns
    ///
    /// The value corresponding to the key, or `None` if the key is not found.
    pub fn lookup(&self, key: &K) -> Option<&V> {
        let index = self.hash(key) % self.table_size;
        let mut current = self.table[index].as_ref();

        while let Some(entry) = current {
            if entry.pair.key == *key {
                return Some(&entry.pair.value);
            }
            current = entry.next.as_ref();
        }

        None
    }

    /// Generate the hash of the key.
    fn hash(&self, key: &K) -> usize {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish() as usize
    }

    /// Remove a value from the hash table.
    ///
    /// # Arguments
    ///
    /// * `key` - The key of the value to remove.
    ///
    /// # Returns
    ///
    /// Returns `true` if the value was removed, or `false` if the specified key was not found in the hash table.
    pub fn remove(&mut self, key: K) -> bool {
        let index = self.hash(&key) % self.table_size;
        let mut rover = &mut self.table[index];
        let mut result = false;

        while let Some(mut entry) = rover.take() {
            if entry.pair.key == key {
                // Remove the entry from the chain
                *rover = entry.next.take();
                // Free the entry
                self.free_entry(entry);
                // Update the count of entries
                self.entries -= 1;
                result = true;
                break;
            }
            *rover = Some(entry);
            rover = &mut rover.as_mut().unwrap().next;
        }

        result
    }
}

/// Structure used to iterate over a hash table.
#[derive(Debug)]
pub struct HashTableIterator<'a, K: 'a + Clone, V: 'a + Clone> {
    hash_table: &'a HashTable<K, V>,
    next_entry: Option<&'a HashTableEntry<K, V>>,
    next_chain: usize,
}

impl<'a, K: 'a + Clone, V: 'a + Clone> HashTableIterator<'a, K, V> {
    /// Determine if there are more keys in the hash table to iterate over.
    ///
    /// # Returns
    /// * `true` if there are more values to iterate over, `false` otherwise.
    pub fn has_more(&self) -> bool {
        self.next_entry.is_some()
    }
}

/// Using a hash table iterator, retrieve the next `HashTablePair`.
///
/// Note: To avoid `HashTableEntry` internal `HashTablePair`
///       from being tampered with, and potentially messing with
///       internal table structure, the function returns a copy
///       of `HashTablePair` stored internally.
///
/// # Returns
///
/// The next `HashTablePair` from the hash table, or `None` if there
/// are no more keys to iterate over.
impl<'a, K: Clone, V: Clone> HashTableIterator<'a, K, V> {
    pub fn next(&mut self) -> Option<HashTablePair<K, V>> {
        let current_entry = self.next_entry?;
        let pair = current_entry.pair.clone();

        if let Some(next_entry) = &current_entry.next {
            self.next_entry = Some(next_entry);
        } else {
            self.next_chain += 1;
            self.next_entry = None;

            while self.next_chain < self.hash_table.table_size {
                if let Some(entry) = &self.hash_table.table[self.next_chain] {
                    self.next_entry = Some(entry);
                    break;
                }
                self.next_chain += 1;
            }
        }

        Some(pair)
    }
}

/// Initialise a `HashTableIterator` to iterate over a hash table.
impl<'a, K: Clone, V: Clone> HashTableIterator<'a, K, V> {
    pub fn new(hash_table: &'a HashTable<K, V>) -> Self {
        let mut iterator = HashTableIterator {
            hash_table,
            next_entry: None,
            next_chain: 0,
        };

        // Find the first entry
        for chain in 0..hash_table.table_size {
            if let Some(entry) = &hash_table.table[chain] {
                iterator.next_entry = Some(entry);
                iterator.next_chain = chain;
                break;
            }
        }

        iterator
    }
}