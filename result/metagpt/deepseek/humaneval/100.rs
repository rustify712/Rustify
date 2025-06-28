pub fn make_a_pile(n: i32) -> Vec<i32> {
    (0..n).map(|i| n + 2 * i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_a_pile() {
        assert_eq!(make_a_pile(3), vec![3, 5, 7]);
        assert_eq!(make_a_pile(5), vec![5, 7, 9, 11, 13]);
    }
}