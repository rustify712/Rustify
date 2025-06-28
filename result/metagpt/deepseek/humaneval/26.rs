use std::collections::HashSet;

pub fn remove_duplicates(numbers: &[i32]) -> Vec<i32> {
    let mut seen_once = HashSet::new();
    let mut seen_twice = HashSet::new();
    
    for &num in numbers {
        if seen_twice.contains(&num) {
            continue;
        }
        
        if seen_once.contains(&num) {
            seen_once.remove(&num);
            seen_twice.insert(num);
        } else {
            seen_once.insert(num);
        }
    }
    
    numbers
        .iter()
        .filter(|&num| !seen_twice.contains(num))
        .copied()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_duplicates() {
        assert_eq!(remove_duplicates(&[1, 2, 3, 2, 4]), vec![1, 3, 4]);
        assert_eq!(remove_duplicates(&[5, 5, 5, 5]), vec![]);
    }
}