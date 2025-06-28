/// Separates a string of parentheses into groups of complete parentheses.
///
/// # Arguments
/// * `paren_string` - A string containing parentheses.
///
/// # Returns
/// A vector of strings, each representing a complete group of parentheses.
fn separate_paren_groups(paren_string: &str) -> Vec<String> {
    let mut all_parens = Vec::new();
    let mut current_paren = String::new();
    let mut level = 0;

    for chr in paren_string.chars() {
        if chr == '(' {
            level += 1;
            current_paren.push(chr);
        } else if chr == ')' {
            level -= 1;
            current_paren.push(chr);
            if level == 0 {
                all_parens.push(current_paren);
                current_paren = String::new();
            }
        }
    }

    all_parens
}