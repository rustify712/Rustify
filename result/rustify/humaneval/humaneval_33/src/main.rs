/// Sorts every third element in the input vector while keeping other elements in their original positions.
///
/// # Arguments
/// * `l` - A slice of integers to be partially sorted.
///
/// # Returns
/// A new vector with every third element sorted.
fn sort_third(l: &[i32]) -> Vec<i32> {
    let mut third: Vec<i32> = l.iter().step_by(3).cloned().collect();
    third.sort();

    let mut out = Vec::with_capacity(l.len());
    for (i, &item) in l.iter().enumerate() {
        if i % 3 == 0 {
            out.push(third[i / 3]);
        } else {
            out.push(item);
        }
    }
    out
}