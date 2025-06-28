/// Calculates the median of a vector of floats.
///
/// # Arguments
///
/// * `l` - A vector of floats.
///
/// # Returns
///
/// Returns `Some(f32)` containing the median if the vector is not empty,
/// otherwise returns `None`.
fn median(mut l: Vec<f32>) -> Option<f32> {
    if l.is_empty() {
        return None;
    }
    l.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let len = l.len();
    if len % 2 == 1 {
        Some(l[len / 2])
    } else {
        Some(0.5 * (l[len / 2] + l[len / 2 - 1]))
    }
}