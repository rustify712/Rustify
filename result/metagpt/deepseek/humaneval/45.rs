pub fn triangle_area(a: f32, h: f32) -> f32 {
    a * h * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_area() {
        assert_eq!(triangle_area(5.0, 3.0), 7.5);
        assert_eq!(triangle_area(2.0, 4.0), 4.0);
    }
}