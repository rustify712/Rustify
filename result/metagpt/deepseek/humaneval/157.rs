pub fn right_angle_triangle(a: f64, b: f64, c: f64) -> bool {
    const EPSILON: f64 = 1e-4;
    (a.powi(2) + b.powi(2) - c.powi(2)).abs() < EPSILON ||
    (a.powi(2) + c.powi(2) - b.powi(2)).abs() < EPSILON ||
    (b.powi(2) + c.powi(2) - a.powi(2)).abs() < EPSILON
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_right_angle_triangle() {
        assert!(right_angle_triangle(3.0, 4.0, 5.0));
        assert!(right_angle_triangle(5.0, 12.0, 13.0));
        assert!(!right_angle_triangle(1.0, 2.0, 3.0));
        assert!(right_angle_triangle(3.0, 5.0, 4.0)); // different order
        assert!(!right_angle_triangle(3.0, 3.0, 3.0)); // equilateral
    }
}