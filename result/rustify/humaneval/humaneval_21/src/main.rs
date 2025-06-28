/// Rescales a vector of floating-point numbers to the range [0, 1].
///
/// # Arguments
///
/// * `numbers` - A vector of floating-point numbers to be rescaled.
///
/// # Returns
///
/// * `Option<Vec<f32>>` - A vector of rescaled numbers, or `None` if the input vector is empty.
fn rescale_to_unit(numbers: Vec<f32>) -> Option<Vec<f32>> {
    if numbers.is_empty() {
        return None;
    }

    let min = numbers.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = numbers.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    if min == max {
        return Some(vec![0.0; numbers.len()]);
    }

    let rescaled_numbers: Vec<f32> = numbers
        .iter()
        .map(|&x| (x - min) / (max - min))
        .collect();

    Some(rescaled_numbers)
}