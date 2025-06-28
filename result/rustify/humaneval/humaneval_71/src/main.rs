/// Calculates the area of a triangle given its three sides.
///
/// # Arguments
///
/// * `a` - The length of the first side of the triangle.
/// * `b` - The length of the second side of the triangle.
/// * `c` - The length of the third side of the triangle.
///
/// # Returns
///
/// Returns `Some(area)` if the sides form a valid triangle, otherwise returns `None`.
fn triangle_area(a: f32, b: f32, c: f32) -> Option<f32> {
    if a + b <= c || a + c <= b || b + c <= a {
        return None;
    }
    let h = (a + b + c) / 2.0;
    let area = (h * (h - a) * (h - b) * (h - c)).sqrt();
    Some(area)
}