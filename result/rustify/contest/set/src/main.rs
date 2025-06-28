use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Internal structure representing an entry in the set.
#[derive(Debug, Clone)]
pub struct SetEntry<T> {
    pub data: T,
    pub next: Option<Box<SetEntry<T>>>,
}

/// Represents a set of values. Created using the `new` function and destroyed using the `drop` trait.
#[derive(Debug)]
pub struct Set<T> {
    table: Vec<Option<Box<SetEntry<T>>>>,
    entries: usize,
    table_size: usize,
    prime_index: usize,
    free_func: Option<fn(T)>, // 添加 free_func 字段
}

impl<T> Set<T> {
    /// Creates a new empty set.
    pub fn new() -> Result<Set<T>, &'static str> {
        let mut set = Set {
            table: Vec::new(),
            entries: 0,
            table_size: 0,
            prime_index: 0,
            free_func: None,
        };

        // Allocate the table
        if set.allocate_table().is_err() {
            return Err("Failed to allocate memory for the hash table.");
        }

        Ok(set)
    }

    /// Query if a particular value is in the set.
    pub fn contains(&self, data: &T) -> bool
        where T: Hash + PartialEq {
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        let index = hasher.finish() as usize % self.table_size;
        let mut rover = self.table[index].as_ref();
        while let Some(entry) = rover {
            if entry.data == *data {
                return true;
            }
            rover = entry.next.as_ref();
        }
        false
    }

    /// Create a vector containing all entries in the set.
    pub fn to_array(&self) -> Vec<T>
        where T: Clone {
        let mut array = Vec::with_capacity(self.entries);
        for entry in &self.table {
            let mut rover = entry.as_ref();
            while let Some(entry) = rover {
                array.push(entry.data.clone());
                rover = entry.next.as_ref();
            }
        }
        array
    }

    /// Frees the memory associated with an entry.
    fn free_entry(&mut self, entry: Box<SetEntry<T>>) {
        // If there is a free function registered, call it to free the data for this entry first
        if let Some(free_func) = self.free_func {
            free_func(entry.data);
        }
        // The entry will be automatically dropped and freed by the Box
    }

    /// Allocates a new hash table for the set.
    ///
    /// This function determines the size of the hash table based on the current
    /// prime index. If the prime index exceeds the predefined prime array,
    /// the table size is set to `entries * 10`.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the allocation was successful, otherwise returns `Err`.
    fn allocate_table(&mut self) -> Result<(), &'static str> {
        // Determine the table size based on the current prime index.
        if self.prime_index < SET_NUM_PRIMES {
            self.table_size = SET_PRIMES[self.prime_index];
        } else {
            self.table_size = self.entries * 10;
        }

        // Allocate the table and initialize to None.
        self.table = Vec::with_capacity(self.table_size);
        for _ in 0..self.table_size {
            self.table.push(None);
        }

        // Check if the allocation was successful.
        if self.table.len() == self.table_size {
            Ok(())
        } else {
            Err("Failed to allocate memory for the hash table.")
        }
    }

    /// Retrieve the number of entries in a set
    pub fn num_entries(&self) -> usize {
        self.entries
    }

    /// Remove a value from the set.
    ///
    /// # Arguments
    ///
    /// * `data` - The value to remove from the set.
    ///
    /// # Returns
    ///
    /// `true` if the value was found and removed from the set, `false` if the value was not found in the set.
    pub fn remove(&mut self, data: T) -> bool
        where T: Hash + PartialEq {
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        let index = hasher.finish() as usize % self.table_size;

        let mut rover = &mut self.table[index];

        while let Some(mut entry) = rover.take() {
            if entry.data == data {
                // Found the entry, unlink it from the chain
                self.entries -= 1;
                self.free_entry(entry);
                return true;
            } else {
                // If not found, put the entry back and continue
                *rover = Some(entry);
                rover = &mut rover.as_mut().unwrap().next;
            }
        }

        false
    }
}

impl<T> Drop for Set<T> {
    fn drop(&mut self) {
        // Implement the drop logic to free all entries
    }
}

/// An object used to iterate over a set.
#[derive(Debug)]
pub struct SetIterator<T> {
    pub set: Option<Box<Set<T>>>,
    pub next_entry: Option<Box<SetEntry<T>>>,
    pub next_chain: usize,
}

impl<T> SetIterator<T> {
    /// Determine if there are more values in the set to iterate over.
    ///
    /// # Returns
    /// - `true` if there are more values to iterate over.
    /// - `false` otherwise.
    pub fn has_more(&self) -> bool {
        self.next_entry.is_some()
    }

    /// Using a set iterator, retrieve the next value from the set.
    /// Returns `None` if no more values are available.
    pub fn next(&mut self) -> Option<T> {
        // Get a mutable reference to the set
        let set = self.set.as_mut()?;

        // No more entries?
        if self.next_entry.is_none() {
            return None;
        }

        // We have the result immediately
        let current_entry = self.next_entry.take()?;
        let result = Some(current_entry.data);

        // Advance next_entry to the next SetEntry in the Set.
        if current_entry.next.is_some() {
            // Use the next value in this chain
            self.next_entry = current_entry.next;
        } else {
            // No more entries in this chain. Search the next chain.
            let mut chain = self.next_chain + 1;
            while chain < set.table_size {
                if let Some(entry) = set.table[chain].take() {
                    self.next_entry = Some(entry);
                    break;
                }
                chain += 1;
            }
            self.next_chain = chain;
        }

        result
    }
}

const SET_NUM_PRIMES: usize = 25;
const SET_PRIMES: [usize; SET_NUM_PRIMES] = [
    5, 13, 23, 47, 97, 193, 389, 769, 1543, 3079, 6151, 12289, 24593, 49157, 98317, 196613, 393241,
    786433, 1572869, 3145739, 6291469, 12582917, 25165843, 50331653, 100663319,
];