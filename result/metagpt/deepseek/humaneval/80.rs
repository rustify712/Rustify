pub fn is_happy(s: &str) -> bool {
    if s.len() < 3 {
        return false;
    }
    
    let chars: Vec<char> = s.chars().collect();
    for i in 2..chars.len() {
        if chars[i] == chars[i-1] || chars[i] == chars[i-2] {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_happy() {
        assert!(!is_happy("aab"));
        assert!(is_happy("abc"));
        assert!(!is_happy("aaa"));
        assert!(!is_happy("ab"));
    }
}