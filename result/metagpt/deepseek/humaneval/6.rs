pub fn parse_nested_parens(paren_string: &str) -> Vec<i32> {
    let mut all_levels = Vec::new();
    let mut level = 0;
    let mut max_level = 0;
    
    for chr in paren_string.chars() {
        match chr {
            '(' => {
                level += 1;
                if level > max_level {
                    max_level = level;
                }
            },
            ')' => {
                level -= 1;
                if level == 0 {
                    all_levels.push(max_level);
                    max_level = 0;
                }
            },
            _ => continue
        }
    }
    
    all_levels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_nested_parens() {
        assert_eq!(parse_nested_parens("(()(()))()"), vec![3, 1]);
        assert_eq!(parse_nested_parens("()()()"), vec![1, 1, 1]);
    }
}