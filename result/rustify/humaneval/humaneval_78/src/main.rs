/// Counts the number of specific characters in a string.
///
/// # Arguments
/// * `num` - A string slice that holds the input string.
///
/// # Returns
/// The count of characters in `num` that are in the set {'2', '3', '5', '7', 'B', 'D'}.
fn hex_key(num: &str) -> usize {
    let key = ['2', '3', '5', '7', 'B', 'D'];
    num.chars().filter(|&c| key.contains(&c)).count()
}