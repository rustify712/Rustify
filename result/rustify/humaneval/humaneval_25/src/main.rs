/// Factorizes a given integer `n` into its prime factors.
///
/// # Arguments
/// * `n` - The integer to factorize.
///
/// # Returns
/// A vector containing all the prime factors of `n`.
fn factorize(n: i32) -> Vec<i32> {
    let mut out = Vec::new();
    let mut n = n;
    let mut i = 2;
    while i * i <= n {
        if n % i == 0 {
            n /= i;
            out.push(i);
            i -= 1;
        }
        i += 1;
    }
    out.push(n);
    out
}