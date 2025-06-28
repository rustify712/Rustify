pub fn sort_array(array: &[i32]) -> Vec<i32> {
    if array.is_empty() {
        return vec![];
    }
    
    let mut sorted = array.to_vec();
    
    if (array.first().unwrap() + array.last().unwrap()) % 2 == 1 {
        sorted.sort_unstable();
    } else {
        sorted.sort_unstable_by(|a, b| b.cmp(a));
    }
    
    sorted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_array() {
        assert_eq!(sort_array(&[1, 3, 2]), vec![1, 2, 3]);
        assert_eq!(sort_array(&[3, 2, 1]), vec![3, 2, 1]);
        assert_eq!(sort_array(&[]), vec![]);
    }
}