/// Checks if there exists a triplet in the given slice that sums to zero.
///
/// # Arguments
///
/// * `l` - A slice of integers to check for triplets.
///
/// # Returns
///
/// Returns `true` if a triplet summing to zero exists, otherwise `false`.
fn triples_sum_to_zero(l: &[i32]) -> bool {
    for i in 0..l.len() {
        for j in i + 1..l.len() {
            for k in j + 1..l.len() {
                if l[i] + l[j] + l[k] == 0 {
                    return true;
                }
            }
        }
    }
    false
}