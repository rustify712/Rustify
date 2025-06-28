/// Checks if `x` is a power of `n`.
///
/// # Arguments
///
/// * `x` - The target integer to check.
/// * `n` - The base integer.
///
/// # Returns
///
/// Returns `true` if `x` is a power of `n`, otherwise returns `false`.
fn is_simple_power(x: i32, n: i32) -> bool {
    let mut p = 1;
    let mut count = 0;
    while p <= x && count < 100 {
        if p == x {
            return true;
        }
        p *= n;
        count += 1;
    }
    false
}