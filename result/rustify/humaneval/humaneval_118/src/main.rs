/// Finds the closest vowel in the given word.
/// A vowel is considered closest if it is surrounded by non-vowel characters.
///
/// # Arguments
/// * `word` - The input string to search for the closest vowel.
///
/// # Returns
/// * `Some(char)` - The closest vowel if found.
/// * `None` - If no such vowel exists.
fn get_closest_vowel(word: &str) -> Option<char> {
    let vowels = "AEIOUaeiou";
    for i in (1..word.len() - 1).rev() {
        let current_char = word.chars().nth(i)?;
        if vowels.contains(current_char) {
            let prev_char = word.chars().nth(i - 1)?;
            let next_char = word.chars().nth(i + 1)?;
            if !vowels.contains(prev_char) && !vowels.contains(next_char) {
                return Some(current_char);
            }
        }
    }
    None
}