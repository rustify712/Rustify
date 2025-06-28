pub fn median(numbers: &mut [f32]) -> f32 {
    numbers.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let len = numbers.len();
    if len % 2 == 1 {
        numbers[len / 2]
    } else {
        0.5 * (numbers[len / 2] + numbers[len / 2 - 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_median() {
        let mut odd = vec![1.0, 3.0, 2.0];
        assert_eq!(median(&mut odd), 2.0);
        
        let mut even = vec![1.0, 3.0, 2.0, 4.0];
        assert_eq!(median(&mut even), 2.5);
    }
}