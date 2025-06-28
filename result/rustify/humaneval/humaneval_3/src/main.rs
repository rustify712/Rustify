/// Checks if the cumulative sum of the operations ever goes below zero.
///
/// # Arguments
///
/// * `operations` - A slice of integers representing the operations.
///
/// # Returns
///
/// Returns `true` if the cumulative sum goes below zero at any point, otherwise returns `false`.
fn below_zero(operations: &[i32]) -> bool {
    let mut num = 0;
    for &operation in operations {
        num += operation;
        if num < 0 {
            return true;
        }
    }
    false
}