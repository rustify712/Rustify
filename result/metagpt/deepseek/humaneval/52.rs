pub fn below_threshold(numbers: &[i32], threshold: i32) -> bool {
    numbers.iter().all(|&x| x < threshold)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_below_threshold() {
        assert!(below_threshold(&[1, 2, 3], 4));
        assert!(!below_threshold(&[1, 2, 4], 4));
    }
}