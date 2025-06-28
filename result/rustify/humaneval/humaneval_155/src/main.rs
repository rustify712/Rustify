/// Counts the number of even and odd digits in the given number.
///
/// # Arguments
/// * `num` - The integer to count even and odd digits in.
///
/// # Returns
/// A tuple where the first element is the count of even digits and the second element is the count of odd digits.
fn even_odd_count(num: i32) -> (usize, usize) {
    let w = num.abs().to_string();
    let mut even_count = 0;
    let mut odd_count = 0;

    for c in w.chars() {
        if c.to_digit(10).unwrap() % 2 == 0 {
            even_count += 1;
        } else {
            odd_count += 1;
        }
    }

    (even_count, odd_count)
}