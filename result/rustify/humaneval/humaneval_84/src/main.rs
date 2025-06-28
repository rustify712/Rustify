/// Converts an integer to a binary string representation of the sum of its digits.
///
/// # Arguments
/// * `n` - The integer to convert.
///
/// # Returns
/// A binary string representation of the sum of the digits of `n`.
fn solve(n: i32) -> String {
    let sum: u32 = n.to_string().chars().map(|c| c.to_digit(10).unwrap()).sum();
    let mut binary = String::new();
    let mut sum = sum;
    while sum > 0 {
        binary = format!("{}{}", sum % 2, binary);
        sum /= 2;
    }
    binary
}