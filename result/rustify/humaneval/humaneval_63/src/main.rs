/// Computes the nth element of a Fibonacci-like sequence where each element is the sum of the previous three.
///
/// # Arguments
///
/// * `n` - The index of the element to compute.
///
/// # Returns
///
/// Returns `Some(value)` if `n` is within the valid range, otherwise returns `None`.
fn fibfib(n: usize) -> Option<i32> {
    if n >= 100 {
        return None;
    }

    let mut ff = vec![0; 100];
    ff[0] = 0;
    ff[1] = 0;
    ff[2] = 1;

    for i in 3..=n {
        ff[i] = ff[i - 1] + ff[i - 2] + ff[i - 3];
    }

    Some(ff[n])
}