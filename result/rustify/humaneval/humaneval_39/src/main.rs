/// Finds the nth prime Fibonacci number.
///
/// # Arguments
///
/// * `n` - The position of the prime Fibonacci number to find.
///
/// # Returns
///
/// Returns `Some(u32)` if the nth prime Fibonacci number is found, otherwise `None`.
fn prime_fib(n: u32) -> Option<u32> {
    let mut f1 = 1;
    let mut f2 = 2;
    let mut count = 0;

    while count < n {
        let next_fib = f1 + f2;
        f1 = f2;
        f2 = next_fib;

        if is_prime(f1) {
            count += 1;
            if count == n {
                return Some(f1);
            }
        }
    }

    None
}

/// Checks if a number is prime.
///
/// # Arguments
///
/// * `num` - The number to check.
///
/// # Returns
///
/// Returns `true` if the number is prime, otherwise `false`.
fn is_prime(num: u32) -> bool {
    if num < 2 {
        return false;
    }
    for i in 2..=(num as f64).sqrt() as u32 {
        if num % i == 0 {
            return false;
        }
    }
    true
}