// set.rs

/// Set implementation in Rust.
///
/// This implementation provides similar functionality to the C version,
/// using a hash table to store unique values.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub struct Set<T> {
    table: Vec<Option<Vec<T>>>,
    entries: usize,
    table_size: usize,
    prime_index: usize,
}

impl<T: Eq + Hash + Clone> Set<T> {
    /// Create a new set.
    ///
    /// # Returns
    /// A new `Set` instance.
    pub fn new() -> Self {
        let initial_size = 193; // Starting with the first prime number
        Set {
            table: vec![None; initial_size],
            entries: 0,
            table_size: initial_size,
            prime_index: 0,
        }
    }

    /// Hash function for the value.
    fn hash(&self, value: &T) -> usize {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        (hasher.finish() % self.table_size as u64) as usize
    }

    /// Insert a value into the set.
    ///
    /// # Arguments
    /// * `value` - The value to insert.
    pub fn insert(&mut self, value: T) {
        let index = self.hash(&value);
        if let Some(bucket) = &mut self.table[index] {
            if !bucket.contains(&value) {
                bucket.push(value);
                self.entries += 1;
            }
        } else {
            self.table[index] = Some(vec![value]);
            self.entries += 1;
        }

        if self.entries > self.table_size {
            self.enlarge();
        }
    }

    /// Check if a value is in the set.
    ///
    /// # Arguments
    /// * `value` - The value to check.
    ///
    /// # Returns
    /// `true` if the value is in the set, `false` otherwise.
    pub fn contains(&self, value: &T) -> bool {
        let index = self.hash(value);
        if let Some(bucket) = &self.table[index] {
            bucket.contains(value)
        } else {
            false
        }
    }

    /// Remove a value from the set.
    ///
    /// # Arguments
    /// * `value` - The value to remove.
    ///
    /// # Returns
    /// `true` if the value was removed, `false` if it was not found.
    pub fn remove(&mut self, value: &T) -> bool {
        let index = self.hash(value);
        if let Some(bucket) = &mut self.table[index] {
            if let Some(pos) = bucket.iter().position(|x| x == value) {
                bucket.swap_remove(pos);
                self.entries -= 1;
                return true;
            }
        }
        false
    }

    /// Enlarge the set when the load factor is too high.
    fn enlarge(&mut self) {
        let new_size = self.table_size * 2;
        let mut new_table = vec![None; new_size];

        for bucket in self.table.drain(..) {
            if let Some(mut bucket) = bucket {
                for value in bucket.drain(..) {
                    let index = self.hash(&value) % new_size;
                    if let Some(new_bucket) = &mut new_table[index] {
                        new_bucket.push(value);
                    } else {
                        new_table[index] = Some(vec![value]);
                    }
                }
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
    fn test_set_operations() {
        let mut set = Set::new();
        set.insert(1);
        set.insert(2);
        set.insert(3);

        assert!(set.contains(&1));
        assert!(set.contains(&2));
        assert!(set.contains(&3));
        assert!(!set.contains(&4));

        assert!(set.remove(&2));
        assert!(!set.contains(&2));
        assert!(!set.remove(&4));
    }
}