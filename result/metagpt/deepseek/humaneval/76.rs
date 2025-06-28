pub fn is_simple_power(x: u32, n: u32) -> bool {
    if n == 1 {
        return x == 1;
    }
    
    let mut p = 1;
    while p <= x {
        if p == x {
            return true;
        }
        p = p.checked_mul(n).unwrap_or(x + 1);
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_simple_power() {
        assert!(is_simple_power(8, 2));  // 2^3
        assert!(is_simple_power(27, 3)); // 3^3
        assert!(!is_simple_power(10, 2));
        assert!(is_simple_power(1, 1));
    }
}