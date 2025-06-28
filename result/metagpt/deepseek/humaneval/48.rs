pub fn is_palindrome(text: &str) -> bool {
    text.chars().eq(text.chars().rev())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_palindrome() {
        assert!(is_palindrome("madam"));
        assert!(!is_palindrome("hello"));
        assert!(is_palindrome(""));
    }
}