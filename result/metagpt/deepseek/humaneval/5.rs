pub fn intersperse(numbers: &[i32], delimiter: i32) -> Vec<i32> {
    if numbers.is_empty() {
        return Vec::new();
    }
    
    let mut result = Vec::with_capacity(2 * numbers.len() - 1);
    result.push(numbers[0]);
    
    for &num in &numbers[1..] {
        result.push(delimiter);
        result.push(num);
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intersperse() {
        assert_eq!(intersperse(&[1, 2, 3], 0), vec![1, 0, 2, 0, 3]);
        assert_eq!(intersperse(&[], 0), vec![]);
    }
}