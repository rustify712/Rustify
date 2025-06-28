pub fn sort_third(l: &[i32]) -> Vec<i32> {
    let mut result = l.to_vec();
    
    // Collect and sort elements at indices divisible by 3
    let mut third_elements: Vec<_> = l.iter()
        .enumerate()
        .filter(|(i, _)| i % 3 == 0)
        .map(|(_, &x)| x)
        .collect();
    
    third_elements.sort_unstable();
    
    // Replace the elements in the result
    for (i, &x) in third_elements.iter().enumerate() {
        result[i * 3] = x;
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_third() {
        assert_eq!(sort_third(&[5, 6, 3, 4, 8, 9, 2]), vec![2, 6, 3, 4, 8, 9, 5]);
        assert_eq!(sort_third(&[1, 2, 3]), vec![1, 2, 3]);
        assert_eq!(sort_third(&[]), vec![]);
    }
}