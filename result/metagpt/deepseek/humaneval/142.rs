pub fn sum_squares(lst: &[i32]) -> i32 {
    lst.iter().enumerate()
        .map(|(i, &x)| {
            if i % 3 == 0 {
                x * x
            } else if i % 4 == 0 {
                x * x * x
            } else {
                x
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_squares() {
        assert_eq!(sum_squares(&[1, 2, 3, 4, 5]), 1 + 2 + 9 + 16 + 5);
        assert_eq!(sum_squares(&[0, 0, 0, 0, 0]), 0);
    }
}