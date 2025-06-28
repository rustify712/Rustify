use std::collections::HashSet;

pub fn count_distinct_characters(s: &str) -> usize {
    s.chars()
        .map(|c| c.to_ascii_lowercase())
        .collect::<HashSet<_>>()
        .len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_distinct_characters() {
        assert_eq!(count_distinct_characters("abcABC"), 3);
        assert_eq!(count_distinct_characters("aabbcc"), 3);
    }
}