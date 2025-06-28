pub fn rescale_to_unit(numbers: &[f32]) -> Vec<f32> {
    if numbers.is_empty() {
        return Vec::new();
    }
    
    let min = numbers.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max = numbers.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    if (max - min).abs() < f32::EPSILON {
        return vec![0.0; numbers.len()];
    }
    
    numbers.iter().map(|&x| (x - min) / (max - min)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rescale_to_unit() {
        assert_eq!(
            rescale_to_unit(&[1.0, 2.0, 3.0, 4.0]),
            vec![0.0, 0.33333334, 0.6666667, 1.0]
        );
        assert_eq!(
            rescale_to_unit(&[5.0, 5.0, 5.0]),
            vec![0.0, 0.0, 0.0]
        );
    }
}