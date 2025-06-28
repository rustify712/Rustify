/// Checks if string `a` contains any rotation of string `b`.
///
/// # Arguments
///
/// * `a` - The string to search within.
/// * `b` - The string whose rotations are to be checked.
///
/// # Returns
///
/// Returns `true` if `a` contains any rotation of `b`, otherwise `false`.
fn cycpattern_check(a: &str, b: &str) -> bool {
    for i in 0..b.len() {
        let (left, right) = b.split_at(i);
        let rotate = format!("{}{}", right, left);
        if a.contains(&rotate) {
            return true;
        }
    }
    false
}