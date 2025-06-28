pub fn prime_length(s: &str) -> bool {
    let l = s.len();
    if l < 2 {
        return false;
    }
    for i in 2..=((l as f64).sqrt() as usize) {
        if l % i == 0 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prime_length() {
        assert!(prime_length("abc"));  // 长度3是质数
        assert!(!prime_length("abcd")); // 长度4不是质数
        assert!(!prime_length("a"));   // 长度1不是质数
    }
}