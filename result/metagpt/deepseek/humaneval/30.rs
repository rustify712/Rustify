pub fn get_positive(numbers: &[f32]) -> Vec<f32> {
    numbers
        .iter()
        .filter(|&&x| x > 0.0)
        .copied()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_positive() {
        assert_eq!(get_positive(&[1.0, -2.0, 3.0, -4.0]), vec![1.0, 3.0]);
        assert_eq!(get_positive(&[]), vec![]);
    }
}