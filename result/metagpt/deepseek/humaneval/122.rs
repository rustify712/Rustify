pub fn add_elements(arr: &[i32], k: usize) -> i32 {
    arr.iter()
        .take(k)
        .filter(|&&x| x >= -99 && x <= 99)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_elements() {
        assert_eq!(add_elements(&[1, 2, 3, 4, 5], 3), 6);
        assert_eq!(add_elements(&[-100, 0, 100], 3), 0);
        assert_eq!(add_elements(&[99, -99], 2), 0);
    }
}