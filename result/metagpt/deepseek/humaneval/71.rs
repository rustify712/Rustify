pub fn triangle_area(a: f32, b: f32, c: f32) -> f32 {
    if a + b <= c || a + c <= b || b + c <= a {
        return -1.0;
    }
    let h = (a + b + c) / 2.0;
    (h * (h - a) * (h - b) * (h - c)).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_area() {
        assert_eq!(triangle_area(3.0, 4.0, 5.0), 6.0);
        assert_eq!(triangle_area(1.0, 1.0, 3.0), -1.0);
    }
}