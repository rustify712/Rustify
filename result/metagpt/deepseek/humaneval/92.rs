pub fn any_int(a: f32, b: f32, c: f32) -> bool {
    if a.fract() != 0.0 || b.fract() != 0.0 || c.fract() != 0.0 {
        return false;
    }
    
    a + b == c || a + c == b || b + c == a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_any_int() {
        assert!(any_int(1.0, 2.0, 3.0));
        assert!(!any_int(1.5, 2.0, 3.5));
        assert!(any_int(5.0, 3.0, 2.0));
    }
}