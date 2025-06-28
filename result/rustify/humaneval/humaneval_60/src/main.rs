/// Calculates the sum of all integers from 1 to n.
///
/// # Arguments
///
/// * `n` - The upper limit of the sum.
///
/// # Returns
///
/// The sum of all integers from 1 to n.
fn sum_to_n(n: i32) -> i32 {
    n * (n + 1) / 2
}