/// Filters and sorts a list of strings, keeping only those with even lengths.
/// The strings are first sorted lexicographically, then by length in ascending order.
///
/// # Arguments
/// * `lst` - A vector of strings to be filtered and sorted.
///
/// # Returns
/// A vector of strings that have even lengths, sorted as described.
fn sorted_list_sum(lst: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = lst.into_iter().filter(|s| s.len() % 2 == 0).collect();
    out.sort_by(|a, b| a.len().cmp(&b.len()).then(a.cmp(b)));
    out
}