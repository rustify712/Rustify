/// Calculates the sum and product of all elements in the given vector.
///
/// # Arguments
///
/// * `numbers` - A slice of integers.
///
/// # Returns
///
/// A tuple containing the sum and product of the elements.
fn sum_product(numbers: &[i32]) -> (i32, i32) {
    let sum = numbers.iter().sum();
    let product = numbers.iter().fold(1, |acc, &x| acc * x);
    (sum, product)
}