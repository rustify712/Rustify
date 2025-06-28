pub fn count_upper(s: &str) -> usize {
    const UVOWEL: &str = "AEIOU";
    s.chars()
        .enumerate()
        .filter(|(i, c)| i % 2 == 0 && UVOWEL.contains(*c))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_upper() {
        assert_eq!(count_upper("aAbBcCdDeE"), 2);
        assert_eq!(count_upper("ABCDEF"), 3);
        assert_eq!(count_upper("xyz"), 0);
    }
}