use std::collections::HashSet;

pub fn same_chars(s0: &str, s1: &str) -> bool {
    let set0: HashSet<char> = s0.chars().collect();
    let set1: HashSet<char> = s1.chars().collect();
    
    set0 == set1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_chars() {
        assert!(same_chars("abc", "cba"));
        assert!(!same_chars("abc", "def"));
        assert!(same_chars("", ""));
    }
}