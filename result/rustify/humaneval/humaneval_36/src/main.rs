/// Counts the numbers between 0 and n-1 that are divisible by 11 or 13 and contain the digit 7.
///
/// # Arguments
/// * `n` - The upper limit of the range to check.
///
/// # Returns
/// The count of numbers that meet the criteria.
fn fizz_buzz(n: i32) -> i32 {
    let mut count = 0;
    for i in 0..n {
        if i % 11 == 0 || i % 13 == 0 {
            let mut q = i;
            while q > 0 {
                if q % 10 == 7 {
                    count += 1;
                }
                q /= 10;
            }
        }
    }
    count
}