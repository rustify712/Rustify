pub fn reverse_delete(s: &str, c: &str) -> (String, bool) {
    // 过滤掉c中包含的字符
    let filtered: String = s.chars()
        .filter(|ch| !c.contains(*ch))
        .collect();
    
    // 检查是否是回文
    let is_palindrome = filtered.chars().eq(filtered.chars().rev());
    
    (filtered, is_palindrome)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse_delete() {
        assert_eq!(reverse_delete("abcde", "ace"), ("bd".to_string(), false));
        assert_eq!(reverse_delete("racecar", "xyz"), ("racecar".to_string(), true));
        assert_eq!(reverse_delete("", "abc"), ("".to_string(), true));
    }
}