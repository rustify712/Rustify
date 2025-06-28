/// Checks if a string is "happy".
///
/// A string is considered "happy" if its length is at least 3 and no character
/// is the same as the previous or the one before that.
///
/// # Arguments
///
/// * `s` - A string slice to check.
///
/// # Returns
///
/// Returns `true` if the string is "happy", otherwise `false`.
fn is_happy(s: &str) -> bool {
    if s.len() < 3 {
        return false;
    }
    for i in 2..s.len() {
        if s.as_bytes()[i] == s.as_bytes()[i - 1] || s.as_bytes()[i] == s.as_bytes()[i - 2] {
            return false;
        }
    }
    true
}