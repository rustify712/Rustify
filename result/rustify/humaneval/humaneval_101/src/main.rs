/// Splits a string into words based on spaces and commas.
///
/// # Arguments
/// * `s` - A string slice that holds the input string.
///
/// # Returns
/// A vector of strings containing the words.
fn words_string(s: &str) -> Vec<String> {
    let mut current = String::new();
    let mut out = Vec::new();
    let s = format!("{} ", s);
    for c in s.chars() {
        if c == ' ' || c == ',' {
            if !current.is_empty() {
                out.push(current);
                current = String::new();
            }
        } else {
            current.push(c);
        }
    }
    out
}