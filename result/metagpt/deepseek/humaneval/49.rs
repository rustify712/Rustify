pub fn modp(n: u32, p: u32) -> u32 {
    if n == 0 {
        return 1;
    }
    let mut result = 1;
    for _ in 0..n {
        result = (result * 2) % p;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modp() {
        assert_eq!(modp(3, 5), 3);
        assert_eq!(modp(1101, 101), 2);
        assert_eq!(modp(0, 101), 1);
        assert_eq!(modp(3, 11), 8);
        assert_eq!(modp(100, 101), 1);
    }
}