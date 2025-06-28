/// Removes characters from `s` that are present in `c` and checks if the resulting string is a palindrome.
///
/// # Arguments
/// * `s` - The input string.
/// * `c` - The string containing characters to be removed from `s`.
///
/// # Returns
/// A tuple containing the filtered string and a boolean indicating if it is a palindrome.
fn reverse_delete(s: &str, c: &str) -> (String, bool) {
    let n: String = s.chars().filter(|&ch| !c.contains(ch)).collect();
    if n.is_empty() {
        return (n, true);
    }
    let is_palindrome = n.chars().rev().collect::<String>() == n;
    (n, is_palindrome)
}