pub fn is_palindrome(s: &str) -> bool {
    s.chars().eq(s.chars().rev())
}

pub fn make_palindrome(s: &str) -> String {
    if s.is_empty() {
        return String::new();
    }
    
    for i in 0..s.len() {
        if is_palindrome(&s[i..]) {
            let prefix = &s[..i];
            return format!("{}{}", s, prefix.chars().rev().collect::<String>());
        }
    }
    
    format!("{}{}", s, s.chars().rev().collect::<String>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_palindrome() {
        assert!(is_palindrome("madam"));
        assert!(!is_palindrome("hello"));
    }

    #[test]
    fn test_make_palindrome() {
        assert_eq!(make_palindrome(""), "");
        assert_eq!(make_palindrome("cat"), "catac");
        assert_eq!(make_palindrome("cata"), "catac");
    }
}