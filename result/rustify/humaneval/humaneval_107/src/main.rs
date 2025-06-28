/// Counts the number of even and odd palindromic numbers up to `n`.
///
/// # Arguments
///
/// * `n` - The upper limit for checking palindromic numbers.
///
/// # Returns
///
/// A tuple containing the count of even palindromic numbers and the count of odd palindromic numbers.
fn even_odd_palindrome(n: i32) -> (i32, i32) {
    let mut num1 = 0;
    let mut num2 = 0;

    for i in 1..=n {
        let w = i.to_string();
        let p: String = w.chars().rev().collect();

        if w == p {
            if i % 2 == 1 {
                num1 += 1;
            } else {
                num2 += 1;
            }
        }
    }

    (num2, num1)
}