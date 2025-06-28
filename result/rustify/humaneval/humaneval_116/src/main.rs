/// Sorts an array of integers based on the number of 1's in their binary representation.
/// If two numbers have the same number of 1's, they are sorted by their value.
///
/// # Arguments
/// * `arr` - A vector of integers to be sorted.
///
/// # Returns
/// A new vector of integers sorted according to the specified criteria.
fn sort_array(arr: Vec<i32>) -> Vec<i32> {
    let mut arr_with_bits: Vec<(i32, i32)> = arr.into_iter().map(|x| {
        let bits = x.abs().count_ones() as i32;
        (x, bits)
    }).collect();

    arr_with_bits.sort_by(|a, b| {
        a.1.cmp(&b.1).then(a.0.cmp(&b.0))
    });

    arr_with_bits.into_iter().map(|(x, _)| x).collect()
}