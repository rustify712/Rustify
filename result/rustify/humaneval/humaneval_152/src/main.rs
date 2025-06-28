/// Compares two vectors of integers and returns a new vector containing the absolute differences.
///
/// # Arguments
///
/// * `game` - A slice of integers representing the first vector.
/// * `guess` - A slice of integers representing the second vector.
///
/// # Returns
///
/// A `Vec<i32>` containing the absolute differences between corresponding elements of `game` and `guess`.
fn compare(game: &[i32], guess: &[i32]) -> Vec<i32> {
    game.iter()
        .zip(guess.iter())
        .map(|(g1, g2)| (g1 - g2).abs())
        .collect()
}