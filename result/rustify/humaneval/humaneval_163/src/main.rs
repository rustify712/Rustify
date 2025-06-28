/// Generates a vector of integers between `a` and `b` that are less than 10 and even.
///
/// # Arguments
///
/// * `a` - The lower bound of the range.
/// * `b` - The upper bound of the range.
///
/// # Returns
///
/// A vector of integers that satisfy the conditions.
fn generate_integers(a: i32, b: i32) -> Vec<i32> {
    let (a, b) = if b < a { (b, a) } else { (a, b) };
    let mut out = Vec::new();
    for i in a..=b {
        if i < 10 && i % 2 == 0 {
            out.push(i);
        }
    }
    out
}