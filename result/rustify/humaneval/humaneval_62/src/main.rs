/// Computes the derivative of a given vector of floats.
///
/// # Arguments
/// * `xs` - A slice of f32 values representing the input vector.
///
/// # Returns
/// A `Vec<f32>` containing the computed derivative values.
fn derivative(xs: &[f32]) -> Vec<f32> {
    xs.iter()
        .enumerate()
        .skip(1)
        .map(|(i, &x)| i as f32 * x)
        .collect()
}