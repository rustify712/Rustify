/// Computes the special factorial of a given number `n`.
/// The special factorial is defined as the product of factorials from 1 to `n`.
///
/// # Arguments
///
/// * `n` - The upper limit for the factorial product.
///
/// # Returns
///
/// The special factorial of `n` as a `u64`.
fn special_factorial(n: u32) -> u64 {
    let mut fact: u64 = 1;
    let mut bfact: u64 = 1;
    for i in 1..=n {
        fact *= i as u64;
        bfact *= fact;
    }
    bfact
}