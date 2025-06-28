use std::collections::HashMap;

pub fn search(lst: &[i32]) -> i32 {
    let mut freq = HashMap::new();
    let mut max = -1;
    
    for &num in lst {
        *freq.entry(num).or_insert(0) += 1;
        let count = freq[&num];
        if count >= num && num > max {
            max = num;
        }
    }
    
    max
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search() {
        assert_eq!(search(&[1, 2, 3, 4, 5]), 1);
        assert_eq!(search(&[2, 2, 3, 3, 3]), 2);
        assert_eq!(search(&[5, 5, 5, 5, 5]), -1);
    }
}