/// Checks if the given integer `n` is even and greater than or equal to 8.
///
/// # Arguments
///
/// * `n` - The integer to check.
///
/// # Returns
///
/// Returns `true` if `n` is even and greater than or equal to 8, otherwise returns `false`.
fn is_equal_to_sum_even(n: i32) -> bool {
    n % 2 == 0 && n >= 8
}