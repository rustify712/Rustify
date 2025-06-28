pub fn is_equal_to_sum_even(n: i32) -> bool {
    n % 2 == 0 && n >= 8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_equal_to_sum_even() {
        assert!(is_equal_to_sum_even(8));
        assert!(!is_equal_to_sum_even(7));
        assert!(!is_equal_to_sum_even(6));
    }
}