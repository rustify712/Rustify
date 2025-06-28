/// Checks if the last character of the given string is a letter and the previous character is not a letter.
///
/// # Arguments
/// * `txt` - A string slice to be checked.
///
/// # Returns
/// * `true` if the last character is a letter and the previous character is not a letter, otherwise `false`.
fn check_if_last_char_is_a_letter(txt: &str) -> bool {
    if txt.is_empty() {
        return false;
    }
    let last_char = txt.chars().last().unwrap();
    if !last_char.is_ascii_alphabetic() {
        return false;
    }
    if txt.len() == 1 {
        return true;
    }
    let second_last_char = txt.chars().nth(txt.len() - 2).unwrap();
    if second_last_char.is_ascii_alphabetic() {
        return false;
    }
    true
}