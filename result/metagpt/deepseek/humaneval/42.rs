pub fn incr_list(numbers: &[i32]) -> Vec<i32> {
    numbers.iter().map(|&x| x + 1).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_incr_list() {
        assert_eq!(incr_list(&[1, 2, 3]), vec![2, 3, 4]);
        assert_eq!(incr_list(&[]), vec![]);
    }
}