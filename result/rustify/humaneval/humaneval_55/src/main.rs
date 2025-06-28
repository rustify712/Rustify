/// Computes the nth Fibonacci number.
///
/// # Arguments
///
/// * `n` - The index of the Fibonacci number to compute.
///
/// # Returns
///
/// The nth Fibonacci number.
fn fib(n: usize) -> u64 {
    let mut f = vec![0; n + 1];
    f[0] = 0;
    if n > 0 {
        f[1] = 1;
    }
    for i in 2..=n {
        f[i] = f[i - 1] + f[i - 2];
    }
    f[n]
}