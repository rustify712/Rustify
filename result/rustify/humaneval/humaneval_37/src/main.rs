/// Sorts the elements at even indices in the given slice and returns a new vector.
/// The elements at odd indices remain unchanged.
///
/// # Arguments
/// * `l` - A slice of `f32` values.
///
/// # Returns
/// A new `Vec<f32>` with even-indexed elements sorted and odd-indexed elements unchanged.
fn sort_even(l: &[f32]) -> Vec<f32> {
    let mut even: Vec<f32> = l.iter().step_by(2).cloned().collect();
    even.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut out = Vec::with_capacity(l.len());
    for (i, &value) in l.iter().enumerate() {
        if i % 2 == 0 {
            out.push(even[i / 2]);
        } else {
            out.push(value);
        }
    }
    out
}