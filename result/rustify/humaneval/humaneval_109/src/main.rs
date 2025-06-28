/// Checks if the array can be made non-decreasing by moving at most one element.
///
/// # Arguments
///
/// * `arr` - A slice of integers representing the array.
///
/// # Returns
///
/// Returns `true` if the array can be made non-decreasing by moving at most one element, otherwise `false`.
fn move_one_ball(arr: &[i32]) -> bool {
    let mut num = 0;
    if arr.is_empty() {
        return true;
    }
    for i in 1..arr.len() {
        if arr[i] < arr[i - 1] {
            num += 1;
        }
    }
    if arr[arr.len() - 1] > arr[0] {
        num += 1;
    }
    num < 2
}