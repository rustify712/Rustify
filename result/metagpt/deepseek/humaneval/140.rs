pub fn fix_spaces(text: &str) -> String {
    let mut result = String::new();
    let mut space_count = 0;
    
    for c in text.chars() {
        if c == ' ' {
            space_count += 1;
        } else {
            match space_count {
                1 => result.push('_'),
                2 => result.push_str("__"),
                n if n > 2 => result.push('-'),
                _ => ()
            }
            space_count = 0;
            result.push(c);
        }
    }
    
    // Handle trailing spaces
    match space_count {
        1 => result.push('_'),
        2 => result.push_str("__"),
        n if n > 2 => result.push('-'),
        _ => ()
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fix_spaces() {
        assert_eq!(fix_spaces("a  b c"), "a--b_c");
        assert_eq!(fix_spaces("  a  b  "), "--a--b__");
        assert_eq!(fix_spaces(""), "");
    }
}