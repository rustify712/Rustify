/// Checks if there are any two elements in the given slice of floats that are closer than the specified threshold.
///
/// # Arguments
///
/// * `numbers` - A slice of floating-point numbers.
/// * `threshold` - The maximum allowed difference between two elements to be considered close.
///
/// # Returns
///
/// Returns `true` if there are any two elements in the slice whose difference is less than the threshold,
/// otherwise returns `false`.
fn has_close_elements(numbers: Vec<f32>, threshold: f32) -> bool {
    for i in 0..numbers.len() {
        for j in i+1..numbers.len() {
            if (numbers[i] - numbers[j]).abs() < threshold {
                return true;
            }
        }
    }
    false
}