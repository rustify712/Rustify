pub fn mean_absolute_deviation(numbers: &[f32]) -> f32 {
    let avg = numbers.iter().sum::<f32>() / numbers.len() as f32;
    numbers.iter().map(|&x| (x - avg).abs()).sum::<f32>() / numbers.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_absolute_deviation() {
        let numbers = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(mean_absolute_deviation(&numbers), 1.2);
    }
}