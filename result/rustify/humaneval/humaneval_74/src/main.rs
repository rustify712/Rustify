/// Compares the total length of strings in two vectors and returns the vector with the smaller total length.
///
/// # Arguments
/// * `lst1` - The first vector of strings.
/// * `lst2` - The second vector of strings.
///
/// # Returns
/// The vector with the smaller total length of strings.
fn total_match(lst1: Vec<String>, lst2: Vec<String>) -> Vec<String> {
    let num1: usize = lst1.iter().map(|s| s.len()).sum();
    let num2: usize = lst2.iter().map(|s| s.len()).sum();
    if num1 > num2 {
        lst2
    } else {
        lst1
    }
}