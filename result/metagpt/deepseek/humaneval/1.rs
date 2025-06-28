pub fn separate_paren_groups(paren_string: &str) -> Vec<String> {
    let mut all_parens = Vec::new();
    let mut current_paren = String::new();
    let mut level = 0;

    for chr in paren_string.chars() {
        if chr == ' ' {
            continue;
        }
        if chr == '(' {
            level += 1;
            current_paren.push(chr);
        }
        if chr == ')' {
            level -= 1;
            current_paren.push(chr);
            if level == 0 {
                all_parens.push(current_paren.clone());
                current_paren.clear();
            }
        }
    }

    all_parens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_separate_paren_groups() {
        assert_eq!(
            separate_paren_groups("(())()(()())"),
            vec!["(())()", "(()())"]
        );
    }
}