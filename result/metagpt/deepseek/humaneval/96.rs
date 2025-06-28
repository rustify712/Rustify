pub fn count_up_to(n: u32) -> Vec<u32> {
    if n < 2 {
        return vec![];
    }
    
    let mut primes = vec![2];
    
    for i in 3..n {
        if primes.iter().take_while(|&&p| p * p <= i).all(|&p| i % p != 0) {
            primes.push(i);
        }
    }
    
    primes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_up_to() {
        assert_eq!(count_up_to(5), vec![2, 3]);
        assert_eq!(count_up_to(10), vec![2, 3, 5, 7]);
        assert_eq!(count_up_to(1), vec![]);
    }
}