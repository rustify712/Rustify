/// Determines if a given integer `a` can be expressed as the product of exactly two prime numbers.
///
/// # Arguments
///
/// * `a` - The integer to check.
///
/// # Returns
///
/// Returns `true` if `a` can be expressed as the product of exactly two prime numbers, otherwise returns `false`.
fn is_multiply_prime(a: i32) -> bool {
    let mut num = 0;
    let mut a = a;
    for i in 2..=((a as f64).sqrt() as i32) {
        while a % i == 0 && a > i {
            a /= i;
            num += 1;
        }
    }
    num == 2
}