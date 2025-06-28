/// Finds the largest negative and smallest positive integers in a list.
///
/// # Arguments
///
/// * `lst` - A slice of integers.
///
/// # Returns
///
/// A tuple containing the largest negative integer and the smallest positive integer.
/// If no negative or positive integers are found, the corresponding value is 0.
fn largest_smallest_integers(lst: &[i32]) -> (i32, i32) {
    let maxneg = lst.iter().filter(|&&x| x < 0).max().copied().unwrap_or(0);
    let minpos = lst.iter().filter(|&&x| x > 0).min().copied().unwrap_or(0);
    (maxneg, minpos)
}