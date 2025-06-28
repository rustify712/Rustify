pub fn digit_sum(s: &str) -> u32 {
    s.chars()
        .filter(|c| c.is_ascii_uppercase())
        .map(|c| c as u32)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_digit_sum() {
        assert_eq!(digit_sum("ABC"), 198); // A(65) + B(66) + C(67) = 198
        assert_eq!(digit_sum("aBc"), 66);
        assert_eq!(digit_sum(""), 0);
    }
}