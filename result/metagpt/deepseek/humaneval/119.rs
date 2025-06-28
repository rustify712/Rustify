pub fn match_parens(lst: [&str; 2]) -> &'static str {
    fn is_balanced(s: &str) -> bool {
        let mut count = 0;
        for c in s.chars() {
            match c {
                '(' => count += 1,
                ')' => {
                    count -= 1;
                    if count < 0 {
                        return false;
                    }
                },
                _ => {}
            }
        }
        count == 0
    }
    
    if is_balanced(&format!("{}{}", lst[0], lst[1])) {
        return "Yes";
    }
    
    if is_balanced(&format!("{}{}", lst[1], lst[0])) {
        return "Yes";
    }
    
    "No"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_parens() {
        assert_eq!(match_parens(["(", ")"]), "Yes");
        assert_eq!(match_parens([")", "("]), "No");
        assert_eq!(match_parens(["(()", ")"]), "Yes");
    }
}