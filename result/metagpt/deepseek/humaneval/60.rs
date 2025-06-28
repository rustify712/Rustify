pub fn sum_to_n(n: u32) -> u32 {
    n * (n + 1) / 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_to_n() {
        assert_eq!(sum_to_n(30), 465);
        assert_eq!(sum_to_n(100), 5050);
        assert_eq!(sum_to_n(5), 15);
        assert_eq!(sum_to_n(10), 55);
        assert_eq!(sum_to_n(1), 1);
    }
}