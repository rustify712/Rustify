/// Checks if any two of the three given floating-point numbers sum to the third.
///
/// # Arguments
///
/// * `a` - A floating-point number.
/// * `b` - A floating-point number.
/// * `c` - A floating-point number.
///
/// # Returns
///
/// Returns `true` if any two of the numbers sum to the third, and all three numbers are integers.
/// Otherwise, returns `false`.
fn any_int(a: f64, b: f64, c: f64) -> bool {
    if a.round() != a || b.round() != b || c.round() != c {
        return false;
    }
    a + b == c || a + c == b || b + c == a
}