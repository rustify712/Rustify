// hash_int.rs

/// Hash function for a pointer to an integer in Rust.
///
/// In Rust, we use references instead of raw pointers for safety.
/// This function takes a reference to an integer and returns its hash value.
///
/// # Arguments
/// * `location` - A reference to the integer.
///
/// # Returns
/// The hash value of the integer.
pub fn int_hash(location: &i32) -> u32 {
    *location as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_hash() {
        let value: i32 = 42;
        let hash_value = int_hash(&value);
        assert_eq!(hash_value, 42u32);

        let negative_value: i32 = -42;
        let hash_negative_value = int_hash(&negative_value);
        assert_eq!(hash_negative_value, (-42i32) as u32);
    }
}