pub fn largest_prime_factor(mut n: u32) -> u32 {
    let mut i = 2;
    while i * i <= n {
        while n % i == 0 && n > i {
            n /= i;
        }
        i += 1;
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_largest_prime_factor() {
        assert_eq!(largest_prime_factor(13195), 29);
        assert_eq!(largest_prime_factor(2048), 2);
    }
}