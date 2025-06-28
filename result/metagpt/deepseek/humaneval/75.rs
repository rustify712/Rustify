pub fn is_multiply_prime(a: u32) -> bool {
    let mut num = 0;
    let mut n = a;
    let mut i = 2;
    
    while i * i <= n {
        while n % i == 0 && n > i {
            n /= i;
            num += 1;
        }
        i += 1;
    }
    
    num == 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_multiply_prime() {
        assert!(is_multiply_prime(6));  // 2*3
        assert!(is_multiply_prime(10)); // 2*5
        assert!(!is_multiply_prime(8));  // 2*2*2
        assert!(!is_multiply_prime(7));  // prime
    }
}