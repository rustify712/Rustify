/// Filters a vector of strings, returning only those that start with the given prefix.
///
/// # Arguments
///
/// * `strings` - A slice of strings to filter.
/// * `prefix` - The prefix to filter by.
///
/// # Returns
///
/// A vector of strings that start with the given prefix.
fn filter_by_prefix(strings: &[String], prefix: &str) -> Vec<String> {
    strings
        .iter()
        .filter(|s| s.starts_with(prefix))
        .cloned()
        .collect()
}