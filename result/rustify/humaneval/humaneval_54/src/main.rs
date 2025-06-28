/// Checks if two strings contain the same set of characters, regardless of order or frequency.
///
/// # Arguments
///
/// * `s0` - The first string to compare.
/// * `s1` - The second string to compare.
///
/// # Returns
///
/// Returns `true` if both strings contain the same set of characters, otherwise `false`.
fn same_chars(s0: &str, s1: &str) -> bool {
    use std::collections::HashSet;
    let set0: HashSet<char> = s0.chars().collect();
    let set1: HashSet<char> = s1.chars().collect();
    set0 == set1
}