pub fn digits(n: i32) -> i32 {
    let s = n.to_string();
    let mut prod = 1;
    let mut has_odd = false;
    
    for c in s.chars() {
        if let Some(d) = c.to_digit(10) {
            if d % 2 == 1 {
                has_odd = true;
                prod *= d as i32;
            }
        }
    }
    
    if has_odd { prod } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_digits() {
        assert_eq!(digits(123), 3); // 1*3=3
        assert_eq!(digits(246), 0); // 无奇数
        assert_eq!(digits(13579), 945); // 1*3*5*7*9=945
    }
}