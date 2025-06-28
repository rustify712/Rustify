pub fn unique(arr: &[i32]) -> Vec<i32> {
    let mut sorted = arr.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    sorted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unique() {
        assert_eq!(unique(&[3, 1, 2, 3, 2]), vec![1, 2, 3]);
        assert_eq!(unique(&[]), vec![]);
    }
}