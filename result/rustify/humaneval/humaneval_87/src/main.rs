/// Returns a vector of tuples representing the indices (i, j) where lst[i][j] == x.
///
/// # Arguments
///
/// * `lst` - A 2D vector of integers.
/// * `x` - The integer to search for.
///
/// # Returns
///
/// A vector of tuples (i, j) where lst[i][j] == x.
fn get_row(lst: &Vec<Vec<i32>>, x: i32) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (i, row) in lst.iter().enumerate() {
        for (j, &value) in row.iter().enumerate().rev() {
            if value == x {
                out.push((i, j));
            }
        }
    }
    out
}