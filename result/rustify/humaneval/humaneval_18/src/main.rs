/// Counts the number of times a substring appears in a string.
///
/// # Arguments
///
/// * `str` - The string to search within.
/// * `substring` - The substring to search for.
///
/// # Returns
///
/// The number of times the substring appears in the string.
fn how_many_times(str: &str, substring: &str) -> usize {
    let mut out = 0;
    if str.is_empty() {
        return 0;
    }
    for i in 0..=str.len().saturating_sub(substring.len()) {
        if str[i..i + substring.len()] == *substring {
            out += 1;
        }
    }
    out
}