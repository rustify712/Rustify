pub fn int_to_mini_roman(number: i32) -> String {
    let mut num = number;
    let mut result = String::new();
    
    let rep = ["m", "cm", "d", "cd", "c", "xc", "l", "xl", "x", "ix", "v", "iv", "i"];
    let values = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1];
    let mut pos = 0;
    
    while num > 0 {
        while num >= values[pos] {
            result.push_str(rep[pos]);
            num -= values[pos];
        }
        if num > 0 {
            pos += 1;
        }
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_to_mini_roman() {
        assert_eq!(int_to_mini_roman(1), "i");
        assert_eq!(int_to_mini_roman(4), "iv");
        assert_eq!(int_to_mini_roman(9), "ix");
        assert_eq!(int_to_mini_roman(42), "xlix");
        assert_eq!(int_to_mini_roman(99), "xcix");
        assert_eq!(int_to_mini_roman(2023), "mmxxiii");
    }
}