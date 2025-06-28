/// Computes a sequence similar to the Fibonacci sequence, where each element is the sum of the previous four elements.
/// The initial conditions are f[0] = 0, f[1] = 0, f[2] = 2, f[3] = 0.
///
/// # Arguments
/// * `n` - The index of the element to compute.
///
/// # Returns
/// The value of the nth element in the sequence.
fn fib4(n: usize) -> i32 {
    let mut f = vec![0; n + 1];
    f[0] = 0;
    f[1] = 0;
    f[2] = 2;
    f[3] = 0;
    for i in 4..=n {
        f[i] = f[i - 1] + f[i - 2] + f[i - 3] + f[i - 4];
    }
    f[n]
}