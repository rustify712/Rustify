pub fn cycpattern_check(a: &str, b: &str) -> bool {
    let len_b = b.len();
    if len_b == 0 {
        return false;
    }
    
    let mut rotate = String::with_capacity(len_b);
    for i in 0..len_b {
        rotate.clear();
        rotate.push_str(&b[i..]);
        rotate.push_str(&b[..i]);
        
        if a.contains(&rotate) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cycpattern_check() {
        assert!(cycpattern_check("abcde", "cdea"));
        assert!(!cycpattern_check("abcde", "ced"));
        assert!(cycpattern_check("aaaa", "a"));
        assert!(!cycpattern_check("abc", ""));
        assert!(cycpattern_check("abab", "ba"));
    }
}