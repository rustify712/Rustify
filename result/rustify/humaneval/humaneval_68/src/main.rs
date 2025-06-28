/// Finds the smallest even number in the array and returns it along with its index.
/// If no even number is found, returns `None`.
///
/// # Arguments
///
/// * `arr` - A slice of integers to search through.
///
/// # Returns
///
/// An `Option` containing a tuple of the smallest even number and its index, or `None` if no even number is found.
fn pluck(arr: &[i32]) -> Option<(i32, usize)> {
    arr.iter()
        .enumerate()
        .filter(|&(_, &x)| x % 2 == 0)
        .min_by_key(|&(_, &x)| x)
        .map(|(i, &x)| (x, i))
}