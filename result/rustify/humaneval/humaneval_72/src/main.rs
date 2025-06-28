/// Checks if the given vector is a palindrome and if the sum of its elements is less than or equal to the given weight.
///
/// # Arguments
///
/// * `q` - A vector of integers to check for palindrome and sum.
/// * `w` - The maximum allowed sum of the vector elements.
///
/// # Returns
///
/// Returns `true` if the vector is a palindrome and the sum of its elements is less than or equal to `w`, otherwise returns `false`.
fn will_it_fly(q: &Vec<i32>, w: i32) -> bool {
    let sum: i32 = q.iter().sum();
    if q.iter().zip(q.iter().rev()).all(|(a, b)| a == b) && sum <= w {
        true
    } else {
        false
    }
}