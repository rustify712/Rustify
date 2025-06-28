/// Inserts a delimiter between elements of a vector.
///
/// # Arguments
///
/// * `numbers` - A slice of integers to be interspersed.
/// * `delimiter` - The integer to insert between elements.
///
/// # Returns
///
/// A new `Vec<i32>` with the delimiter inserted between elements.
fn intersperse(numbers: &[i32], delimiter: i32) -> Vec<i32> {
    let mut out = Vec::with_capacity(numbers.len() * 2 - 1);
    if let Some(&first) = numbers.first() {
        out.push(first);
    }
    for &num in numbers.iter().skip(1) {
        out.push(delimiter);
        out.push(num);
    }
    out
}