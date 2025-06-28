/// Checks if the given string is a palindrome.
///
/// # Arguments
///
/// * `text` - A string slice that holds the text to be checked.
///
/// # Returns
///
/// Returns `true` if the text is a palindrome, otherwise `false`.
fn is_palindrome(text: &str) -> bool {
    let reversed: String = text.chars().rev().collect();
    reversed == text
}