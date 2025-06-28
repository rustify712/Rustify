/// Calculates the sum of elements in the vector based on their indices.
/// If the index is divisible by 3, the element is squared before adding.
/// If the index is divisible by 4, the element is cubed before adding.
/// Otherwise, the element is added as is.
///
/// # Arguments
///
/// * `lst` - A slice of integers to be processed.
///
/// # Returns
///
/// The calculated sum as an integer.
fn sum_squares(lst: &[i32]) -> i32 {
    let mut sum = 0;
    for (i, &num) in lst.iter().enumerate() {
        if i % 3 == 0 {
            sum += num * num;
        } else if i % 4 == 0 {
            sum += num * num * num;
        } else {
            sum += num;
        }
    }
    sum
}