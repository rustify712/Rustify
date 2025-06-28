/// Computes the rolling maximum of a given slice of integers.
///
/// # Arguments
///
/// * `numbers` - A slice of integers.
///
/// # Returns
///
/// A vector of integers where each element is the maximum value up to that index in the input slice.
fn rolling_max(numbers: &[i32]) -> Vec<i32> {
    let mut out = Vec::new();
    let mut max = i32::MIN;
    for &num in numbers {
        if num > max {
            max = num;
        }
        out.push(max);
    }
    out
}