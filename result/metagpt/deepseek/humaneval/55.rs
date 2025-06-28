pub fn fib(n: usize) -> u64 {
    if n == 0 {
        return 0;
    }
    let (mut a, mut b) = (0, 1);
    for _ in 1..n {
        let c = a + b;
        a = b;
        b = c;
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fib() {
        assert_eq!(fib(10), 55);
        assert_eq!(fib(1), 1);
        assert_eq!(fib(8), 21);
    }
}