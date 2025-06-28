pub fn has_close_elements(numbers: &[f32], threshold: f32) -> bool {
    numbers.iter().enumerate().any(|(i, &x)| {
        numbers[i+1..].iter().any(|&y| (x - y).abs() < threshold)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_has_close_elements() {
        assert!(has_close_elements(&[1.0, 2.0, 3.0, 4.0, 1.5], 0.5));
        assert!(!has_close_elements(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.5));
    }
}