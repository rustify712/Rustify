// hash_string.rs

/// String hash functions in Rust.
///
/// These functions provide similar functionality to the C functions for
/// hashing strings, including case-insensitive hashing.

/// Hash function for a string.
///
/// This is the djb2 string hash function.
///
/// # Arguments
/// * `string` - A reference to the string.
///
/// # Returns
/// The hash value of the string.
pub fn string_hash(string: &str) -> u32 {
    let mut result = 5381u32;
    for &byte in string.as_bytes() {
        result = (result << 5).wrapping_add(result).wrapping_add(byte as u32);
    }
    result
}

/// Hash function for a string, ignoring case.
///
/// This is a case-insensitive version of the djb2 string hash function.
///
/// # Arguments
/// * `string` - A reference to the string.
///
/// # Returns
/// The hash value of the string, ignoring case.
pub fn string_nocase_hash(string: &str) -> u32 {
    let mut result = 5381u32;
    for &byte in string.as_bytes() {
        result = (result << 5)
            .wrapping_add(result)
            .wrapping_add(byte.to_ascii_lowercase() as u32);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_hash() {
        let hash_value = string_hash("hello");
        assert_eq!(hash_value, 210714636441u32 as u32); // Expected value based on djb2

        let hash_value_empty = string_hash("");
        assert_eq!(hash_value_empty, 5381u32); // djb2 for empty string
    }

    #[test]
    fn test_string_nocase_hash() {
        let hash_value = string_nocase_hash("Hello");
        assert_eq!(hash_value, string_nocase_hash("hello")); // Case-insensitive check

        let hash_value_empty = string_nocase_hash("");
        assert_eq!(hash_value_empty, 5381u32); // djb2 for empty string
    }
}