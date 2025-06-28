/// Counts the number of odd digits in each string of the input vector and returns a vector of formatted strings.
///
/// # Arguments
///
/// * `lst` - A slice of strings to process.
///
/// # Returns
///
/// A vector of strings, each containing the count of odd digits in the corresponding input string.
fn odd_count(lst: &[String]) -> Vec<String> {
    lst.iter()
        .map(|s| {
            let count = s.chars()
                .filter(|&c| c.is_ascii_digit() && (c as u8) % 2 == 1)
                .count();
            format!("the number of odd elements in the string {} of the input.", count)
        })
        .collect()
}