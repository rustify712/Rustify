
/// Hash function used to generate hash values for values inserted into a
/// bloom filter.
///
/// # Arguments
///
/// * `data` - The value to generate a hash value for.
///
/// # Returns
///
/// The hash value.
type BloomFilterHashFunc<T> = fn(T) -> u32;

/// A bloom filter structure.
pub struct BloomFilter<T> {
    /// Hash function used to generate hash values for values inserted into the bloom filter.
    pub hash_func: fn(T) -> u32,
    /// The table storing the bits of the bloom filter.
    pub table: Vec<u8>,
    /// The size of the bloom filter table.
    pub table_size: usize,
    /// The number of hash functions to apply to each element on insertion.
    pub num_functions: usize,
}

const SALTS: [u32; 64] = [
    0x00000001, 0x00000002, 0x00000004, 0x00000008,
    0x00000010, 0x00000020, 0x00000040, 0x00000080,
    0x00000100, 0x00000200, 0x00000400, 0x00000800,
    0x00001000, 0x00002000, 0x00004000, 0x00008000,
    0x00010000, 0x00020000, 0x00040000, 0x00080000,
    0x00100000, 0x00200000, 0x00400000, 0x00800000,
    0x01000000, 0x02000000, 0x04000000, 0x08000000,
    0x10000000, 0x20000000, 0x40000000, 0x80000000,
    0x1b97206a, 0x372e40d4, 0x6e5c81a8, 0xdcb90350,
    0xb97206a0, 0x72e40d40, 0xe5c81a80, 0xcb903500,
    0x97206a00, 0x2e40d400, 0x5c81a800, 0xb9035000,
    0x7206a000, 0xe40d4000, 0xc81a8000, 0x90350000,
    0x206a0000, 0x40d40000, 0x81a80000, 0x03500000,
    0x06a00000, 0x0d400000, 0x1a800000, 0x35000000,
    0x6a000000, 0xd4000000, 0xa8000000, 0x50000000,
    0xa0000000, 0x40000000, 0x80000000, 0x00000000,
];

impl<T> BloomFilter<T> {
    /// Create a new bloom filter.
    ///
    /// # Arguments
    ///
    /// * `table_size` - The size of the bloom filter. The greater the table size, the more elements can be stored, and the lesser the chance of false positives.
    /// * `hash_func` - Hash function to use on values stored in the filter.
    /// * `num_functions` - Number of hash functions to apply to each element on insertion. This running time for insertion and queries is proportional to this value. The more functions applied, the lesser the chance of false positives. The maximum number of functions is 64.
    ///
    /// # Returns
    ///
    /// A new bloom filter, or an error if it was not possible to allocate memory.
    pub fn new(table_size: usize, hash_func: fn(T) -> u32, num_functions: usize) -> Result<BloomFilter<T>, &'static str> {
        // There is a limit on the number of functions which can be applied, due to the table size
        if num_functions > 64 {
            return Err("Number of hash functions exceeds the maximum limit.");
        }

        // Allocate table, each entry is one bit; these are packed into bytes. When allocating we must round the length up to the nearest byte.
        let table = vec![0; (table_size + 7) / 8];

        Ok(BloomFilter {
            hash_func,
            table,
            table_size,
            num_functions,
        })
    }

    /// Insert a value into the bloom filter.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to insert.
    pub fn insert(&mut self, value: T) {
        let hash = (self.hash_func)(value);
        for i in 0..self.num_functions {
            let subhash = hash ^ SALTS[i];
            let index = subhash % self.table_size as u32;
            let byte_index = (index / 8) as usize;
            let bit_index = (index % 8) as u8;
            self.table[byte_index] |= 1 << bit_index;
        }
    }

    /// Query if a value is possibly in the bloom filter.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to query.
    ///
    /// # Returns
    ///
    /// True if the value is possibly in the filter, false if it is definitely not.
    pub fn query(&self, value: T) -> bool {
        let hash = (self.hash_func)(value);
        for i in 0..self.num_functions {
            let subhash = hash ^ SALTS[i];
            let index = subhash % self.table_size as u32;
            let byte_index = (index / 8) as usize;
            let bit_index = (index % 8) as u8;
            if self.table[byte_index] & (1 << bit_index) == 0 {
                return false;
            }
        }
        true
    }

    /// Read the current state of the bloom filter.
    ///
    /// # Returns
    ///
    /// A vector representing the current state of the bloom filter.
    pub fn read(&self) -> Vec<u8> {
        self.table.clone()
    }

    /// Load a state into the bloom filter.
    ///
    /// # Arguments
    ///
    /// * `state` - The state to load into the bloom filter.
    pub fn load(&mut self, state: Vec<u8>) {
        self.table = state;
    }
}

/// Find the union of two bloom filters. Values are present in the resulting filter if they are present in either of the original filters.
/// Both of the original filters must have been created using the same parameters to `new`.
///
/// # Arguments
///
/// * `filter1` - The first filter.
/// * `filter2` - The second filter.
///
/// # Returns
///
/// A new filter which is the union of the two filters, or an error if the two filters specified were created with different parameters.
pub fn union<T>(filter1: &BloomFilter<T>, filter2: &BloomFilter<T>) -> Result<BloomFilter<T>, &'static str> {
    // To perform this operation, both filters must be created with the same values.
    if filter1.table_size != filter2.table_size || filter1.num_functions != filter2.num_functions || filter1.hash_func as usize != filter2.hash_func as usize {
        return Err("The two filters must have the same parameters.");
    }

    // Create a new bloom filter for the result
    let mut result = BloomFilter::new(filter1.table_size, filter1.hash_func, filter1.num_functions)?;

    // The table is an array of bits, packed into bytes. Round up to the nearest byte.
    let array_size = (filter1.table_size + 7) / 8;

    // Populate the table of the new filter
    for i in 0..array_size {
        result.table[i] = filter1.table[i] | filter2.table[i];
    }

    Ok(result)
}

/// Find the intersection of two bloom filters. Values are only ever
/// present in the resulting filter if they are present in both of the
/// original filters.
///
/// Both of the original filters must have been created using the
/// same parameters to `new`.
///
/// # Arguments
///
/// * `filter1` - The first filter.
/// * `filter2` - The second filter.
///
/// # Returns
///
/// A new filter which is an intersection of the two filters, or an error
/// if the two filters specified were created with different parameters.
pub fn intersection<T>(filter1: &BloomFilter<T>, filter2: &BloomFilter<T>) -> Result<BloomFilter<T>, &'static str> {
    // To perform this operation, both filters must be created with
    // the same values.
    if filter1.table_size != filter2.table_size ||
       filter1.num_functions != filter2.num_functions ||
       filter1.hash_func != filter2.hash_func {
        return Err("Filters were created with different parameters.");
    }

    // Create a new bloom filter for the result
    let mut result = BloomFilter::new(filter1.table_size, filter1.hash_func, filter1.num_functions)?;

    // The table is an array of bits, packed into bytes.  Round up
    // to the nearest byte.
    let array_size = (filter1.table_size + 7) / 8;

    // Populate the table of the new filter
    for i in 0..array_size {
        result.table[i] = filter1.table[i] & filter2.table[i];
    }

    Ok(result)
}