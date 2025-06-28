pub fn circular_shift(x: i32, shift: usize) -> String {
    let mut s = x.to_string();
    let len = s.len();
    
    if len == 0 {
        return s;
    }
    
    let actual_shift = shift % len;
    if actual_shift == 0 {
        return s;
    }
    
    let split_point = len - actual_shift;
    let mut rotated = String::with_capacity(len);
    rotated.push_str(&s[split_point..]);
    rotated.push_str(&s[..split_point]);
    
    rotated
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circular_shift() {
        assert_eq!(circular_shift(12, 1), "21");
        assert_eq!(circular_shift(12345, 2), "45123");
        assert_eq!(circular_shift(123, 0), "123");
        assert_eq!(circular_shift(123, 3), "123");
    }
}