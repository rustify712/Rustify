/// Solves the problem of converting uppercase to lowercase and vice versa.
/// If the string contains no letters, returns the reversed string.
///
/// # Arguments
/// * `s` - A string slice that holds the input string.
///
/// # Returns
/// A `String` containing the transformed or reversed string.
fn solve(s: &str) -> String {
    let mut nletter = 0;
    let mut out = String::new();

    for c in s.chars() {
        let w = if c.is_ascii_uppercase() {
            c.to_ascii_lowercase()
        } else if c.is_ascii_lowercase() {
            c.to_ascii_uppercase()
        } else {
            nletter += 1;
            c
        };
        out.push(w);
    }

    if nletter == s.len() {
        s.chars().rev().collect()
    } else {
        out
    }
}