/// Calculates the distribution of fruits based on the given string and total number.
///
/// # Arguments
///
/// * `s` - A string slice that contains the numbers to be extracted.
/// * `n` - The total number of fruits.
///
/// # Returns
///
/// Returns the result of `n - num1 - num2`, where `num1` and `num2` are the numbers extracted from `s`.
/// If the string does not contain valid numbers, returns `None`.
fn fruit_distribution(s: &str, n: i32) -> Option<i32> {
    let mut num1 = String::new();
    let mut num2 = String::new();
    let mut is12 = false;

    for c in s.chars() {
        if c.is_ascii_digit() {
            if !is12 {
                num1.push(c);
            } else {
                num2.push(c);
            }
        } else if !is12 && !num1.is_empty() {
            is12 = true;
        }
    }

    let num1 = num1.parse::<i32>().ok()?;
    let num2 = num2.parse::<i32>().ok()?;

    Some(n - num1 - num2)
}