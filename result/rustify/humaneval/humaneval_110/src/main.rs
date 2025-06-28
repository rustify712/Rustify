/// Determines if the number of even numbers in `lst1` and `lst2` is greater than or equal to the size of `lst1`.
///
/// # Arguments
///
/// * `lst1` - A vector of integers.
/// * `lst2` - Another vector of integers.
///
/// # Returns
///
/// Returns "YES" if the number of even numbers in both vectors is greater than or equal to the size of `lst1`,
/// otherwise returns "NO".
fn exchange(lst1: &Vec<i32>, lst2: &Vec<i32>) -> &'static str {
    let num: usize = lst1.iter().filter(|&x| x % 2 == 0).count() + lst2.iter().filter(|&x| x % 2 == 0).count();
    if num >= lst1.len() {
        "YES"
    } else {
        "NO"
    }
}