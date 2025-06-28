/// Counts the number of distinct characters in a string, ignoring case.
///
/// # Arguments
///
/// * `s` - A string slice that holds the input string.
///
/// # Returns
///
/// The number of distinct characters in the string.
fn count_distinct_characters(s: &str) -> usize {
    let mut distinct_chars = std::collections::HashSet::new();
    for c in s.chars() {
        distinct_chars.insert(c.to_ascii_lowercase());
    }
    distinct_chars.len()
}
