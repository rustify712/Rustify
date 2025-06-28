pub fn special_factorial(n: u32) -> u64 {
    let mut fact: u64 = 1;
    let mut bfact: u64 = 1;
    
    for i in 1..=n {
        fact *= i as u64;
        bfact *= fact;
    }
    
    bfact
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_factorial() {
        assert_eq!(special_factorial(1), 1);
        assert_eq!(special_factorial(2), 2); // 1! * 2! = 1 * 2 = 2
        assert_eq!(special_factorial(3), 12); // 1! * 2! * 3! = 1 * 2 * 6 = 12
    }
}