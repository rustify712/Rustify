pub fn double_the_difference(lst: &[f64]) -> i64 {
    lst.iter()
        .filter(|&&x| {
            let rounded = x.round();
            (x - rounded).abs() < 1e-4 && rounded > 0.0 && rounded as i64 % 2 == 1
        })
        .map(|&x| (x.round() as i64).pow(2))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double_the_difference() {
        assert_eq!(double_the_difference(&[1.0, 2.0, 3.0]), 10); // 1^2 + 3^2 = 10
        assert_eq!(double_the_difference(&[1.1, 2.2, 3.0]), 9); // 3^2 = 9
        assert_eq!(double_the_difference(&[-1.0, 0.0, 1.0]), 1); // 1^2 = 1
    }
}