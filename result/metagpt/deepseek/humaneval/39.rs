pub fn prime_fib(n: usize) -> i32 {
    let mut fib = (1, 2);
    let mut count = 0;
    
    loop {
        fib = (fib.1, fib.0 + fib.1);
        
        if is_prime(fib.0) {
            count += 1;
            if count == n {
                return fib.0;
            }
        }
    }
}

fn is_prime(num: i32) -> bool {
    if num <= 1 {
        return false;
    }
    for i in 2..=(num as f64).sqrt() as i32 {
        if num % i == 0 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prime_fib() {
        assert_eq!(prime_fib(1), 2);
        assert_eq!(prime_fib(2), 3);
        assert_eq!(prime_fib(3), 5);
    }
}