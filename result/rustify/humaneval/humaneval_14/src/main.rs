/// Returns a vector containing all prefixes of the input string.
///
/// # Arguments
///
/// * `s` - The input string.
///
/// # Returns
///
/// A vector of strings, each representing a prefix of the input string.
fn all_prefixes(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for c in s.chars() {
        current.push(c);
        out.push(current.clone());
    }
    out
}