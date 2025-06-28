pub fn rolling_max(numbers: &[i32]) -> Vec<i32> {
    let mut result = Vec::with_capacity(numbers.len());
    let mut current_max = i32::MIN;
    
    for &num in numbers {
        if num > current_max {
            current_max = num;
        }
        result.push(current_max);
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_max() {
        assert_eq!(rolling_max(&[1, 3, 2, 5, 4]), vec![1, 3, 3, 5, 5]);
        assert_eq!(rolling_max(&[-2, -1, -3]), vec![-2, -1, -1]);
    }
}