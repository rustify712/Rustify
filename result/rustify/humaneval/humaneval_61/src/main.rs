/// Checks if the brackets in the given string are correctly matched.
///
/// # Arguments
/// * `brackets` - A string slice containing the brackets to be checked.
///
/// # Returns
/// * `true` if the brackets are correctly matched, `false` otherwise.
fn correct_bracketing(brackets: &str) -> bool {
    let mut level = 0;
    for c in brackets.chars() {
        match c {
            '(' => level += 1,
            ')' => level -= 1,
            _ => {}
        }
        if level < 0 {
            return false;
        }
    }
    level == 0
}