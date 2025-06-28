/// Parses a string containing nested parentheses and returns a vector of integers
/// representing the maximum nesting depth of each complete pair of parentheses.
///
/// # Arguments
///
/// * `paren_string` - A string slice containing nested parentheses.
///
/// # Returns
///
/// A vector of integers where each integer represents the maximum nesting depth
/// of a complete pair of parentheses.
fn parse_nested_parens(paren_string: &str) -> Vec<i32> {
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
            }
            ')' => {
                level -= 1;
                if level == 0 {
                    all_levels.push(max_level);
                    max_level = 0;
                }
            }
            _ => {}
        }
    }

    all_levels
}