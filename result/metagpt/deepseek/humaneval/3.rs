pub fn below_zero(operations: &[i32]) -> bool {
    let mut num = 0;
    for &op in operations {
        num += op;
        if num < 0 {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_below_zero() {
        assert!(below_zero(&[1, -2, 3, -4]));
        assert!(!below_zero(&[1, 2, 3, 4]));
    }
}