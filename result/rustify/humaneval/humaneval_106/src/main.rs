/// Computes a vector of alternating sums and products.
///
/// # Arguments
///
/// * `n` - The upper limit of the range [1, n].
///
/// # Returns
///
/// A vector containing alternating sums and products for each integer in the range [1, n].
fn f(n: i32) -> Vec<i32> {
    let mut sum = 0;
    let mut prod = 1;
    let mut out = Vec::new();

    for i in 1..=n {
        sum += i;
        prod *= i;
        if i % 2 == 0 {
            out.push(prod);
        } else {
            out.push(sum);
        }
    }

    out
}