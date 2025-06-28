pub fn add(lst: &[i32]) -> i32 {
    lst.iter()
        .enumerate()
        .filter(|(i, &x)| i % 2 == 1 && x % 2 == 0)
        .map(|(_, &x)| x)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(&[1, 2, 3, 4, 5, 6]), 6); // 2 + 4 + 6 = 12
        assert_eq!(add(&[1, 3, 5, 7]), 0);
        assert_eq!(add(&[0, 1, 0, 1, 0]), 0);
    }
}