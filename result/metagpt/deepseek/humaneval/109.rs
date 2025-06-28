pub fn move_one_ball(arr: &[i32]) -> bool {
    if arr.is_empty() {
        return true;
    }
    
    let mut num = 0;
    for i in 1..arr.len() {
        if arr[i] < arr[i-1] {
            num += 1;
        }
    }
    
    if arr.last() > arr.first() {
        num += 1;
    }
    
    num < 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_move_one_ball() {
        assert!(move_one_ball(&[1, 2, 3]));
        assert!(move_one_ball(&[3, 1, 2]));
        assert!(!move_one_ball(&[3, 2, 1]));
    }
}