pub fn change_base(mut x: i32, base: i32) -> String {
    if x == 0 {
        return "0".to_string();
    }
    
    let mut result = String::new();
    while x > 0 {
        let digit = x % base;
        result.insert(0, std::char::from_digit(digit as u32, base as u32).unwrap());
        x /= base;
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_change_base() {
        assert_eq!(change_base(10, 2), "1010");
        assert_eq!(change_base(15, 16), "f");
        assert_eq!(change_base(0, 2), "0");
    }
}