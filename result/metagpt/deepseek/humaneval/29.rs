pub fn filter_by_prefix(strings: &[&str], prefix: &str) -> Vec<String> {
    strings
        .iter()
        .filter(|s| s.starts_with(prefix))
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_by_prefix() {
        let strings = ["apple", "banana", "appetizer", "orange"];
        assert_eq!(
            filter_by_prefix(&strings, "app"),
            vec!["apple", "appetizer"]
        );
        assert_eq!(
            filter_by_prefix(&strings, "ora"),
            vec!["orange"]
        );
    }
}