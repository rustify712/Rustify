/// Splits the input string into words based on spaces or commas. If neither is found,
/// counts the number of lowercase even ASCII characters and returns the count as a string.
///
/// # Arguments
/// * `txt` - The input string to be processed.
///
/// # Returns
/// A vector of words if the string contains spaces or commas, otherwise a string
/// representing the count of lowercase even ASCII characters.
fn split_words(txt: &str) -> Vec<String> {
    if txt.contains(' ') {
        return txt.split(' ')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
    }
    if txt.contains(',') {
        return txt.split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
    }
    let count = txt.chars()
        .filter(|&c| c.is_ascii_lowercase() && (c as u8) % 2 == 0)
        .count();
    vec![count.to_string()]
}