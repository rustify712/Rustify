pub fn string_length(s: &str) -> usize {
    s.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_length() {
        assert_eq!(string_length("hello"), 5);
        assert_eq!(string_length(""), 0);
    }
}