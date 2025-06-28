pub fn maximum(mut arr: Vec<i32>, k: usize) -> Vec<i32> {
    arr.sort_unstable();
    arr.iter().rev().take(k).copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maximum() {
        assert_eq!(maximum(vec![1, 2, 3, 4, 5], 3), vec![5, 4, 3]);
        assert_eq!(maximum(vec![5, 4, 3, 2, 1], 2), vec![5, 4]);
        assert_eq!(maximum(vec![1, 1, 1], 1), vec![1]);
    }
}