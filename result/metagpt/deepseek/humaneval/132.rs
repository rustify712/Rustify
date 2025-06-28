pub fn is_nested(s: &str) -> bool {
    let mut count = 0;
    let mut maxcount = 0;
    
    for c in s.chars() {
        match c {
            '[' => count += 1,
            ']' => count -= 1,
            _ => {}
        }
        
        if count < 0 {
            count = 0;
        }
        
        if count > maxcount {
            maxcount = count;
        }
        
        if count <= maxcount - 2 {
            return true;
        }
    }
    
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_nested() {
        assert!(is_nested("[[][]]"));
        assert!(!is_nested("[][]"));
        assert!(!is_nested("[]"));
    }
}