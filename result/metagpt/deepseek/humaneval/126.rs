pub fn is_sorted(lst: &[i32]) -> bool {
    if lst.len() < 2 {
        return true;
    }
    
    for i in 1..lst.len() {
        if lst[i] < lst[i-1] {
            return false;
        }
        if i >= 2 && lst[i] == lst[i-1] && lst[i] == lst[i-2] {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_sorted() {
        assert!(is_sorted(&[1, 2, 3]));
        assert!(!is_sorted(&[1, 2, 2, 2]));
        assert!(!is_sorted(&[3, 2, 1]));
    }
}