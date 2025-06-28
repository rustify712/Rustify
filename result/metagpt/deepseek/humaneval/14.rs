pub fn all_prefixes(s: &str) -> Vec<String> {
    (1..=s.len())
        .map(|i| s[..i].to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_prefixes() {
        assert_eq!(
            all_prefixes("abc"),
            vec!["a", "ab", "abc"]
        );
        assert_eq!(
            all_prefixes(""),
            Vec::<String>::new()
        );
    }
}