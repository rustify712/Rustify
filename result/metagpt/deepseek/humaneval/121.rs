pub fn solutions(lst: &[i32]) -> i32 {
    lst.iter()
        .enumerate()
        .filter(|(i, &x)| i % 2 == 0 && x % 2 == 1)
        .map(|(_, &x)| x)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solutions() {
        assert_eq!(solutions(&[1, 2, 3, 4, 5]), 6); // 1 + 3 + 5 = 9
        assert_eq!(solutions(&[2, 4, 6]), 0);
        assert_eq!(solutions(&[1, 3, 5]), 9);
    }
}