pub fn largest_divisor(n: i32) -> i32 {
    (2..=n/2).rev()
        .find(|&i| n % i == 0)
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_largest_divisor() {
        assert_eq!(largest_divisor(15), 5);
        assert_eq!(largest_divisor(17), 1);
    }
}