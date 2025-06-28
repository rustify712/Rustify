/// Calculates the sum of squares of the ceiling of each element in the list.
///
/// # Arguments
///
/// * `lst` - A slice of f32 values.
///
/// # Returns
///
/// The sum of squares of the ceiling of each element as an i32.
fn sum_squares(lst: &[f32]) -> i32 {
    lst.iter()
        .map(|&x| x.ceil() as i32)
        .map(|x| x * x)
        .sum()
}