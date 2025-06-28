/// Finds the second smallest element in a list of integers.
///
/// # Arguments
///
/// * `lst` - A slice of integers.
///
/// # Returns
///
/// Returns `Some(i32)` if a second smallest element is found, otherwise returns `None`.
fn next_smallest(lst: &[i32]) -> Option<i32> {
    let mut sorted_lst = lst.to_vec();
    sorted_lst.sort();
    for i in 1..sorted_lst.len() {
        if sorted_lst[i] != sorted_lst[i - 1] {
            return Some(sorted_lst[i]);
        }
    }
    None
}