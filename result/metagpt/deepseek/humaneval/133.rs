pub fn sum_squares(lst: &[f64]) -> i32 {
    lst.iter()
        .map(|&x| x.ceil() as i32)
        .map(|x| x * x)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_squares() {
        assert_eq!(sum_squares(&[1.1, 2.2, 3.3]), 29); // 2^2 + 3^2 + 4^2 = 4 + 9 + 16 = 29
        assert_eq!(sum_squares(&[-1.5, 0.5]), 2); // (-1)^2 + 1^2 = 1 + 1 = 2
    }
}