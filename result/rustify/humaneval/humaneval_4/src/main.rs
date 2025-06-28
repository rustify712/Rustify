/// Calculates the mean absolute deviation (MAD) of a slice of floating-point numbers.
///
/// # Arguments
///
/// * `numbers` - A slice of floating-point numbers (`f64`).
///
/// # Returns
///
/// Returns `Some(f64)` representing the mean absolute deviation if the slice is not empty.
/// Returns `None` if the slice is empty.
fn mean_absolute_deviation(numbers: &[f64]) -> Option<f64> {
    if numbers.is_empty() {
        return None;
    }

    let sum: f64 = numbers.iter().sum();
    let avg = sum / numbers.len() as f64;
    let msum: f64 = numbers.iter().map(|&x| (x - avg).abs()).sum();
    Some(msum / numbers.len() as f64)
}