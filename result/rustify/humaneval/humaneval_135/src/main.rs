/// Returns the maximum index `i` where `arr[i] <= i`.
/// If no such index exists, returns `-1`.
///
/// # Arguments
///
/// * `arr` - A slice of integers to check.
///
/// # Returns
///
/// The maximum index `i` where `arr[i] <= i`, or `-1` if no such index exists.
fn can_arrange(arr: &[i32]) -> i32 {
    let mut max = -1;
    for (i, &value) in arr.iter().enumerate() {
        if value <= i as i32 {
            max = i as i32;
        }
    }
    max
}