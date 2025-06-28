/// Checks if the length of the given string is a prime number.
///
/// # Arguments
/// * `s` - A string slice to check the length of.
///
/// # Returns
/// * `true` if the length is a prime number, `false` otherwise.
fn prime_length(s: &str) -> bool {
    let l = s.len();
    if l < 2 {
        return false;
    }
    for i in 2..=(l as f64).sqrt() as usize {
        if l % i == 0 {
            return false;
        }
    }
    true
}