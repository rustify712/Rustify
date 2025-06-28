/// Counts the number of sentences starting with 'I' followed by a space.
///
/// # Arguments
///
/// * `s` - A string slice to be analyzed.
///
/// # Returns
///
/// The number of sentences starting with 'I' followed by a space.
fn is_bored(s: &str) -> usize {
    let mut is_start = true;
    let mut is_i = false;
    let mut sum = 0;

    for c in s.chars() {
        if c == ' ' && is_i {
            is_i = false;
            sum += 1;
        }
        if c == 'I' && is_start {
            is_i = true;
        } else {
            is_i = false;
        }
        if c != ' ' {
            is_start = false;
        }
        if c == '.' || c == '?' || c == '!' {
            is_start = true;
        }
    }

    sum
}