/// Counts the number of vowels in a given string.
///
/// # Arguments
///
/// * `s` - A string slice that holds the input string.
///
/// # Returns
///
/// The number of vowels in the string.
fn vowels_count(s: &str) -> usize {
    let vowels = "aeiouAEIOU";
    let mut count = 0;
    for c in s.chars() {
        if vowels.contains(c) {
            count += 1;
        }
    }
    if s.ends_with('y') || s.ends_with('Y') {
        count += 1;
    }
    count
}