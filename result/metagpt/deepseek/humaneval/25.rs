pub fn factorize(n: i32) -> Vec<i32> {
    let mut factors = Vec::new();
    let mut num = n;
    
    let mut i = 2;
    while i * i <= num {
        while num % i == 0 {
            factors.push(i);
            num /= i;
        }
        i += 1;
    }
    
    if num > 1 {
        factors.push(num);
    }
    
    factors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factorize() {
        assert_eq!(factorize(12), vec![2, 2, 3]);
        assert_eq!(factorize(17), vec![17]);
    }
}