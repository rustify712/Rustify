// bloom_filter.rs

/// Bloom Filter implementation in Rust.
///
/// This implementation provides similar functionality to the C version,
/// using a hash function and multiple hash functions to determine membership.

use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

pub struct BloomFilter<T> {
    table: Vec<u8>,
    table_size: usize,
    num_functions: usize,
    hash_func: fn(&T) -> u64,
}

impl<T: Hash> BloomFilter<T> {
    /// Create a new bloom filter.
    ///
    /// # Arguments
    /// * `table_size` - The size of the table in bits.
    /// * `hash_func` - The hash function to use.
    /// * `num_functions` - The number of hash functions to use.
    ///
    /// # Returns
    /// A new `BloomFilter` instance.
    pub fn new(table_size: usize, hash_func: fn(&T) -> u64, num_functions: usize) -> Self {
        let byte_size = (table_size + 7) / 8;
        BloomFilter {
            table: vec![0; byte_size],
            table_size,
            num_functions,
            hash_func,
        }
    }

    /// Insert a value into the bloom filter.
    ///
    /// # Arguments
    /// * `value` - The value to insert.
    pub fn insert(&mut self, value: &T) {
        let hash = (self.hash_func)(value);
        for i in 0..self.num_functions {
            let subhash = hash ^ SALTS[i];
            let index = (subhash % self.table_size as u64) as usize;
            self.table[index / 8] |= 1 << (index % 8);
        }
    }

    /// Query if a value is in the bloom filter.
    ///
    /// # Arguments
    /// * `value` - The value to query.
    ///
    /// # Returns
    /// `true` if the value is possibly in the filter, `false` if it is definitely not.
    pub fn query(&self, value: &T) -> bool {
        let hash = (self.hash_func)(value);
        for i in 0..self.num_functions {
            let subhash = hash ^ SALTS[i];
            let index = (subhash % self.table_size as u64) as usize;
            if (self.table[index / 8] & (1 << (index % 8))) == 0 {
                return false;
            }
        }
        true
    }
}

const SALTS: [u64; 64] = [
    0x1953c322, 0x588ccf17, 0x64bf600c, 0xa6be3f3d,
    0x341a02ea, 0x15b03217, 0x3b062858, 0x5956fd06,
    0x18b5624f, 0xe3be0b46, 0x20ffcd5c, 0xa35dfd2b,
    0x1fc4a9bf, 0x57c45d5c, 0xa8661c4a, 0x4f1b74d2,
    0x5a6dde13, 0x3b18dac6, 0x05a8afbf, 0xbbda2fe2,
    0xa2520d78, 0xe7934849, 0xd541bc75, 0x09a55b57,
    0x9b345ae2, 0xfc2d26af, 0x38679cef, 0x81bd1e0d,
    0x654681ae, 0x4b3d87ad, 0xd5ff10fb, 0x23b32f67,
    0xafc7e366, 0xdd955ead, 0xe7c34b1c, 0xfeace0a6,
    0xeb16f09d, 0x3c57a72d, 0x2c8294c5, 0xba92662a,
    0xcd5b2d14, 0x743936c8, 0x2489beff, 0xc6c56e00,
    0x74a4f606, 0xb244a94a, 0x5edfc423, 0xf1901934,
    0x24af7691, 0xf6c98b25, 0xea25af46, 0x76d5f2e6,
    0x5e33cdf2, 0x445eb357, 0x88556bd2, 0x70d1da7a,
    0x54449368, 0x381020bc, 0x1c0520bf, 0xf7e44942,
    0xa27e2a58, 0x66866fc5, 0x12519ce7, 0x437a8456,
];

#[cfg(test)]
mod tests {
    use super::*;

    fn hash_value(value: &i32) -> u64 {
        let mut hasher = DefaultHasher::new();
        value.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn test_bloom_filter() {
        let mut filter = BloomFilter::new(1024, hash_value, 3);

        filter.insert(&10);
        filter.insert(&20);
        filter.insert(&30);

        assert!(filter.query(&10));
        assert!(filter.query(&20));
        assert!(filter.query(&30));
        assert!(!filter.query(&40));
    }
}