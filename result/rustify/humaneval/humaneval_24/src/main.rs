/// Finds the largest divisor of `n` that is less than `n`.
///
/// # Arguments
///
/// * `n` - The integer to find the largest divisor for.
///
/// # Returns
///
/// The largest divisor of `n` that is less than `n`. If `n` is less than 2, returns 1.
fn largest_divisor(n: i32) -> i32 {
    for i in 2..=((n as f64).sqrt() as i32) {
        if n % i == 0 {
            return n / i;
        }
    }
    1
}