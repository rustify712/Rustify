pub fn max_fill(grid: Vec<Vec<i32>>, capacity: i32) -> i32 {
    grid.iter()
        .map(|row| {
            let sum: i32 = row.iter().sum();
            if sum > 0 {
                (sum + capacity - 1) / capacity
            } else {
                0
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_fill() {
        assert_eq!(max_fill(vec![vec![1, 2], vec![3, 4]], 2), 4);
        assert_eq!(max_fill(vec![vec![0, 0], vec![0, 0]], 2), 0);
        assert_eq!(max_fill(vec![vec![5]], 3), 2);
    }
}