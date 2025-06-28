/// Multiplies the last digits of the absolute values of two integers.
///
/// # Arguments
///
/// * `a` - The first integer.
/// * `b` - The second integer.
///
/// # Returns
///
/// The product of the last digits of the absolute values of `a` and `b`.
fn multiply(a: i32, b: i32) -> i32 {
    (a.abs() % 10) * (b.abs() % 10)
}