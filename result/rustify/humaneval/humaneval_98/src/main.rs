/// Counts the number of uppercase vowels at even indices in the given string.
///
/// # Arguments
/// * `s` - A string slice containing the input string.
///
/// # Returns
/// The count of uppercase vowels at even indices.
fn count_upper(s: &str) -> usize {
    let uvowel = ['A', 'E', 'I', 'O', 'U'];
    s.chars()
        .enumerate()
        .filter(|&(i, c)| i % 2 == 0 && uvowel.contains(&c))
        .count()
}