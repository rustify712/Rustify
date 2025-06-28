/// Determines if the intersection length of two intervals is a prime number.
///
/// # Arguments
/// * `interval1` - The first interval as a slice of integers.
/// * `interval2` - The second interval as a slice of integers.
///
/// # Returns
/// A `String` indicating whether the intersection length is a prime number ("YES") or not ("NO").
fn intersection(interval1: &[i32], interval2: &[i32]) -> String {
    let inter1 = std::cmp::max(interval1[0], interval2[0]);
    let inter2 = std::cmp::min(interval1[1], interval2[1]);
    let l = inter2 - inter1;

    if l < 2 {
        return "NO".to_string();
    }

    for i in 2..=(l as f64).sqrt() as i32 {
        if l % i == 0 {
            return "NO".to_string();
        }
    }

    "YES".to_string()
}