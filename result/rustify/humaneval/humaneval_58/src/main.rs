/// Returns a vector containing the common elements of two input vectors.
/// The elements in the output vector are unique and sorted in ascending order.
///
/// # Arguments
///
/// * `l1` - A slice of integers representing the first vector.
/// * `l2` - A slice of integers representing the second vector.
///
/// # Returns
///
/// A `Vec<i32>` containing the common elements of `l1` and `l2`.
fn common(l1: &[i32], l2: &[i32]) -> Vec<i32> {
    use std::collections::HashSet;

    let mut out: Vec<i32> = Vec::new();
    let set: HashSet<_> = l2.iter().collect();

    for &item in l1 {
        if !out.contains(&item) && set.contains(&item) {
            out.push(item);
        }
    }

    out.sort();
    out
}