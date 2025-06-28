/// Determines if the given sides form a right-angled triangle.
///
/// # Arguments
///
/// * `a` - The length of the first side.
/// * `b` - The length of the second side.
/// * `c` - The length of the third side.
///
/// # Returns
///
/// Returns `true` if the sides form a right-angled triangle, otherwise `false`.
fn right_angle_triangle(a: f32, b: f32, c: f32) -> bool {
    (a * a + b * b - c * c).abs() < 1e-4 ||
    (a * a + c * c - b * b).abs() < 1e-4 ||
    (b * b + c * c - a * a).abs() < 1e-4
}