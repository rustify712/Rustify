/// Calculates the product of all odd digits in the given number.
/// If there are no odd digits, returns 0.
///
/// # Arguments
/// * `n` - The integer to process.
///
/// # Returns
/// The product of all odd digits in `n`, or 0 if there are no odd digits.
fn digits(n: i32) -> i32 {
    let s = n.to_string();
    let mut prod = 1;
    let mut has_odd = false;

    for c in s.chars() {
        if let Some(digit) = c.to_digit(10) {
            if digit % 2 == 1 {
                has_odd = true;
                prod *= digit as i32;
            }
        }
    }

    if has_odd {
        prod
    } else {
        0
    }
}