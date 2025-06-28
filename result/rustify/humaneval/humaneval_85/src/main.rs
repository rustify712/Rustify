/// Adds all even numbers at odd indices in the given vector.
///
/// # Arguments
///
/// * `lst` - A vector of integers.
///
/// # Returns
///
/// The sum of all even numbers at odd indices.
fn add(lst: &Vec<i32>) -> i32 {
    lst.iter()
        .enumerate()
        .filter(|&(i, &x)| i % 2 == 1 && x % 2 == 0)
        .map(|(_, &x)| x)
        .sum()
}