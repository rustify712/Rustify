/// Checks if the given string has nested square brackets.
///
/// # Arguments
///
/// * `s` - A string slice to check for nested brackets.
///
/// # Returns
///
/// Returns `true` if the string has nested brackets, otherwise `false`.
fn is_nested(s: &str) -> bool {
    let mut count = 0;
    let mut max_count = 0;

    for c in s.chars() {
        match c {
            '[' => count += 1,
            ']' => count -= 1,
            _ => (),
        }

        if count < 0 {
            count = 0;
        }

        if count > max_count {
            max_count = count;
        }

        if count <= max_count - 2 {
            return true;
        }
    }

    false
}