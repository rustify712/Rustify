pub fn filter_by_substring(strings: &[&str], substring: &str) -> Vec<String> {
    strings
        .iter()
        .filter(|s| s.contains(substring))
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_by_substring() {
        let strings = ["hello", "world", "rust", "programming"];
        assert_eq!(
            filter_by_substring(&strings, "o"),
            vec!["hello", "world", "programming"]
        );
    }
}