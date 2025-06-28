/// Checks if a list of integers is sorted in ascending order and does not contain three consecutive equal elements.
///
/// # Arguments
///
/// * `lst` - A slice of integers to be checked.
///
/// # Returns
///
/// Returns `true` if the list is sorted and does not contain three consecutive equal elements, otherwise `false`.
fn is_sorted(lst: &[i32]) -> bool {
    for i in 1..lst.len() {
        if lst[i] < lst[i - 1] {
            return false;
        }
        if i >= 2 && lst[i] == lst[i - 1] && lst[i] == lst[i - 2] {
            return false;
        }
    }
    true
}