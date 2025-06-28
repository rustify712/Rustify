pub fn next_smallest(mut lst: Vec<i32>) -> Option<i32> {
    if lst.len() <= 1 {
        return None;
    }
    
    lst.sort_unstable();
    
    for i in 1..lst.len() {
        if lst[i] != lst[i - 1] {
            return Some(lst[i]);
        }
    }
    
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_smallest() {
        assert_eq!(next_smallest(vec![1, 2, 3]), Some(2));
        assert_eq!(next_smallest(vec![1, 1, 1]), None);
        assert_eq!(next_smallest(vec![1]), None);
    }
}