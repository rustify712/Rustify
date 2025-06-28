pub fn multiply(a: i32, b: i32) -> i32 {
    (a.abs() % 10) * (b.abs() % 10)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multiply() {
        assert_eq!(multiply(123, 456), 18); // 3*6=18
        assert_eq!(multiply(-123, 456), 18);
        assert_eq!(multiply(123, -456), 18);
    }
}