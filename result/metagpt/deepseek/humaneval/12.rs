pub fn longest(strings: &[&str]) -> Option<&str> {
    strings.iter().max_by_key(|s| s.len()).copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_longest() {
        assert_eq!(longest(&["a", "bb", "ccc"]), Some("ccc"));
        assert_eq!(longest(&[]), None);
    }
}