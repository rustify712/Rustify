pub fn decimal_to_binary(decimal: i32) -> String {
    if decimal == 0 {
        return "db0db".to_string();
    }
    
    let mut binary = String::new();
    let mut num = decimal;
    
    while num > 0 {
        binary.push(if num % 2 == 0 { '0' } else { '1' });
        num /= 2;
    }
    
    let binary_str: String = binary.chars().rev().collect();
    format!("db{}db", binary_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decimal_to_binary() {
        assert_eq!(decimal_to_binary(0), "db0db");
        assert_eq!(decimal_to_binary(5), "db101db");
        assert_eq!(decimal_to_binary(10), "db1010db");
    }
}