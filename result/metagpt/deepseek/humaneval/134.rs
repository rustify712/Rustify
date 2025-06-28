pub fn check_if_last_char_is_a_letter(txt: &str) -> bool {
    if txt.is_empty() {
        return false;
    }
    
    let last_char = txt.chars().last().unwrap();
    if !last_char.is_ascii_alphabetic() {
        return false;
    }
    
    if txt.len() == 1 {
        return true;
    }
    
    let prev_char = txt.chars().rev().nth(1).unwrap();
    !prev_char.is_ascii_alphabetic()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_if_last_char_is_a_letter() {
        assert!(check_if_last_char_is_a_letter("a"));
        assert!(!check_if_last_char_is_a_letter("ab"));
        assert!(!check_if_last_char_is_a_letter("a1"));
        assert!(!check_if_last_char_is_a_letter(""));
    }
}