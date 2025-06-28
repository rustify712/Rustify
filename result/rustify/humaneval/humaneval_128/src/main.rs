/// Computes the product of the sum of absolute values and the sign product of the elements in the array.
/// If the array is empty, returns `None`.
///
/// # Arguments
///
/// * `arr` - A slice of integers.
///
/// # Returns
///
/// * `Some(i32)` - The product of the sum of absolute values and the sign product.
/// * `None` - If the array is empty.
fn prod_signs(arr: &[i32]) -> Option<i32> {
    if arr.is_empty() {
        return None;
    }
    let mut sum = 0;
    let mut prods = 1;
    for &num in arr {
        sum += num.abs();
        if num == 0 {
            prods = 0;
        } else if num < 0 {
            prods = -prods;
        }
    }
    Some(sum * prods)
}