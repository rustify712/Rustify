pub fn compare(game: &[i32], guess: &[i32]) -> Vec<i32> {
    game.iter()
        .zip(guess.iter())
        .map(|(&g, &h)| (g - h).abs())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare() {
        assert_eq!(compare(&[1, 2, 3], &[4, 5, 6]), vec![3, 3, 3]);
        assert_eq!(compare(&[10, 20], &[5, 25]), vec![5, 5]);
    }
}