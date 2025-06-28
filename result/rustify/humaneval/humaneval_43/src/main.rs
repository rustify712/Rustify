/// Checks if there exists a pair of distinct elements in the given slice that sum to zero.
///
/// # Arguments
///
/// * `l` - A slice of integers to check for pairs that sum to zero.
///
/// # Returns
///
/// Returns `true` if such a pair exists, otherwise returns `false`.
fn pairs_sum_to_zero(l: &[i32]) -> bool {
    l.iter().enumerate().any(|(i, &x)| l[i+1..].iter().any(|&y| x + y == 0))
}