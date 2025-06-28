pub fn derivative(xs: &[f32]) -> Vec<f32> {
    xs.iter()
        .enumerate()
        .skip(1)
        .map(|(i, &x)| i as f32 * x)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derivative() {
        assert_eq!(derivative(&[1.0, 2.0, 3.0]), vec![2.0, 6.0]);
        assert_eq!(derivative(&[0.0, 1.0, 0.0, 2.0]), vec![1.0, 0.0, 6.0]);
    }
}