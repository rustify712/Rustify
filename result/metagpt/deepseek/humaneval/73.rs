pub fn smallest_change(arr: &[i32]) -> usize {
    let mut changes = 0;
    let len = arr.len();
    
    for i in 0..len/2 {
        if arr[i] != arr[len - 1 - i] {
            changes += 1;
        }
    }
    
    changes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smallest_change() {
        assert_eq!(smallest_change(&[1, 2, 3, 2, 2]), 1);
        assert_eq!(smallest_change(&[1, 2, 3, 4, 5]), 2);
        assert_eq!(smallest_change(&[1, 2, 1]), 0);
    }
}