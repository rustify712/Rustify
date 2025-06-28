// compare_string.rs

/// Comparison functions for strings in Rust.
/// 
/// These functions provide similar functionality to the C functions for
/// comparing strings, including case-insensitive comparisons.

/// Checks if two strings are equal.
///
/// # Arguments
/// * `string1` - A reference to the first string.
/// * `string2` - A reference to the second string.
///
/// # Returns
/// `true` if the strings are equal, `false` otherwise.
pub fn string_equal(string1: &str, string2: &str) -> bool {
    string1 == string2
}

/// Compares two strings lexicographically.
///
/// # Arguments
/// * `string1` - A reference to the first string.
/// * `string2` - A reference to the second string.
///
/// # Returns
/// - `-1` if `string1` is less than `string2`
/// - `1` if `string1` is greater than `string2`
/// - `0` if both are equal
pub fn string_compare(string1: &str, string2: &str) -> i32 {
    string1.cmp(string2) as i32
}

/// Checks if two strings are equal, ignoring case.
///
/// # Arguments
/// * `string1` - A reference to the first string.
/// * `string2` - A reference to the second string.
///
/// # Returns
/// `true` if the strings are equal ignoring case, `false` otherwise.
pub fn string_nocase_equal(string1: &str, string2: &str) -> bool {
    string1.eq_ignore_ascii_case(string2)
}

/// Compares two strings lexicographically, ignoring case.
///
/// # Arguments
/// * `string1` - A reference to the first string.
/// * `string2` - A reference to the second string.
///
/// # Returns
/// - `-1` if `string1` is less than `string2` ignoring case
/// - `1` if `string1` is greater than `string2` ignoring case
/// - `0` if both are equal ignoring case
pub fn string_nocase_compare(string1: &str, string2: &str) -> i32 {
    string1.to_lowercase().cmp(&string2.to_lowercase()) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_equal() {
        assert!(string_equal("hello", "hello"));
        assert!(!string_equal("hello", "world"));
    }

    #[test]
    fn test_string_compare() {
        assert_eq!(string_compare("apple", "banana"), -1);
        assert_eq!(string_compare("banana", "apple"), 1);
        assert_eq!(string_compare("apple", "apple"), 0);
    }

    #[test]
    fn test_string_nocase_equal() {
        assert!(string_nocase_equal("Hello", "hello"));
        assert!(!string_nocase_equal("Hello", "world"));
    }

    #[test]
    fn test_string_nocase_compare() {
        assert_eq!(string_nocase_compare("apple", "Banana"), -1);
        assert_eq!(string_nocase_compare("Banana", "apple"), 1);
        assert_eq!(string_nocase_compare("Apple", "apple"), 0);
    }
}