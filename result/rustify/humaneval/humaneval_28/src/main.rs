/// Concatenates a slice of strings into a single String.
///
/// # Arguments
///
/// * `strings` - A slice of strings to concatenate.
///
/// # Returns
///
/// A new `String` containing the concatenated result.
fn concatenate(strings: &[String]) -> String {
    let mut out = String::new();
    for s in strings {
        out.push_str(s);
    }
    out
}