/// Finds the pair of elements in the given vector with the smallest difference.
///
/// # Arguments
/// * `numbers` - A vector of floating-point numbers.
///
/// # Returns
/// An `Option<(f32, f32)>` containing the pair of elements with the smallest difference,
/// or `None` if the vector contains fewer than 2 elements.
fn find_closest_elements(numbers: Vec<f32>) -> Option<(f32, f32)> {
    if numbers.len() < 2 {
        return None;
    }

    let mut closest_pair: Option<(f32, f32)> = None;
    let mut min_diff = f32::INFINITY;

    for i in 0..numbers.len() {
        for j in (i + 1)..numbers.len() {
            let diff = (numbers[i] - numbers[j]).abs();
            if diff < min_diff {
                min_diff = diff;
                closest_pair = Some((numbers[i], numbers[j]));
            }
        }
    }

    closest_pair.map(|(a, b)| if a < b { (a, b) } else { (b, a) })
}