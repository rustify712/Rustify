/// Computes (2^n) % p.
///
/// # Arguments
/// * `n` - The exponent.
/// * `p` - The modulus.
///
/// # Returns
/// The result of (2^n) % p.
fn modp(n: i32, p: i32) -> i32 {
    let mut out = 1;
    for _ in 0..n {
        out = (out * 2) % p;
    }
    out
}