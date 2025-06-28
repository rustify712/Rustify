/// Generate a hash key from a string, ignoring the case of letters.
///
/// # Arguments
///
/// * `string` - A string slice to generate the hash key from.
///
/// # Returns
///
/// A hash key for the string.
pub fn string_nocase_hash(string: &str) -> u32 {
    let mut result: u32 = 5381;

    for c in string.chars() {
        result = (result << 5).wrapping_add(result).wrapping_add(c.to_lowercase().next().unwrap() as u32);
    }

    result
}

/// Generate a hash key from a string.
///
/// # Arguments
///
/// * `string` - The string to hash.
///
/// # Returns
///
/// A hash key for the string.
pub fn string_hash(string: &str) -> u32 {
    let mut result: u32 = 5381;
    for c in string.chars() {
        result = (result << 5).wrapping_add(result).wrapping_add(c as u32);
    }
    result
}
