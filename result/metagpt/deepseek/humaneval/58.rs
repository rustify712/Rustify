use std::collections::HashSet;

pub fn common(l1: &[i32], l2: &[i32]) -> Vec<i32> {
    let set1: HashSet<_> = l1.iter().cloned().collect();
    let set2: HashSet<_> = l2.iter().cloned().collect();
    
    let mut result: Vec<_> = set1.intersection(&set2).cloned().collect();
    result.sort();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_common() {
        assert_eq!(common(&[1, 2, 3], &[2, 3, 4]), vec![2, 3]);
        assert_eq!(common(&[1, 2, 2, 3], &[2, 2, 3, 4]), vec![2, 3]);
        assert!(common(&[1], &[2]).is_empty());
    }
}