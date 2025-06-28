/// Checks if the given integer is a perfect cube.
///
/// # Arguments
///
/// * `a` - The integer to check.
///
/// # Returns
///
/// Returns `true` if `a` is a perfect cube, otherwise `false`.
fn iscuber(a: i32) -> bool {
    for i in 0..=a.abs() {
        if i * i * i == a.abs() {
            return true;
        }
        if i * i * i > a.abs() {
            break;
        }
    }
    false
}