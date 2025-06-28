/// Finds the longest string in a slice of strings.
///
/// # Arguments
///
/// * `strings` - A slice of strings to search through.
///
/// # Returns
///
/// Returns an `Option<String>` containing the longest string if the slice is not empty,
/// otherwise returns `None`.
fn longest(strings: Vec<String>) -> Option<String> {
    strings
        .into_iter()
        .reduce(|acc, item| {
            if item.len() > acc.len() {
                item
            } else {
                acc
            }
        })
}