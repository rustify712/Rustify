pub fn how_many_times(s: &str, sub: &str) -> usize {
    if s.is_empty() || sub.is_empty() {
        return 0;
    }
    
    s.matches(sub).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_how_many_times() {
        assert_eq!(how_many_times("abcabc", "abc"), 2);
        assert_eq!(how_many_times("aaaa", "aa"), 3);
        assert_eq!(how_many_times("", "abc"), 0);
    }
}