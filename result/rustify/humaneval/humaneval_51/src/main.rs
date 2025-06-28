/// Removes all vowels from the input string.
///
/// # Arguments
/// * `text` - A string slice that holds the input text.
///
/// # Returns
/// A new `String` with all vowels removed.
fn remove_vowels(text: &str) -> String {
    let vowels = "AEIOUaeiou";
    text.chars()
        .filter(|c| !vowels.contains(*c))
        .collect()
}