/// Calculates the minimum number of fills required to fill each row of the grid
/// given a specific capacity.
///
/// # Arguments
///
/// * `grid` - A 2D vector of integers representing the grid.
/// * `capacity` - The capacity of each fill operation.
///
/// # Returns
///
/// The minimum number of fills required.
fn max_fill(grid: &Vec<Vec<i32>>, capacity: i32) -> i32 {
    let mut out = 0;
    for row in grid {
        let sum: i32 = row.iter().sum();
        if sum > 0 {
            out += (sum - 1) / capacity + 1;
        }
    }
    out
}