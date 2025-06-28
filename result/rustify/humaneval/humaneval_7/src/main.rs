/// Filters a vector of strings, returning only those that contain the specified substring.
///
/// # Arguments
///
/// * `strings` - A slice of strings to filter.
/// * `substring` - The substring to search for within each string.
///
/// # Returns
///
/// A vector of strings that contain the specified substring.
fn filter_by_substring(strings: &[String], substring: &str) -> Vec<String> {
    strings
        .iter()
        .filter(|s| s.contains(substring))
        .cloned()
        .collect()
}