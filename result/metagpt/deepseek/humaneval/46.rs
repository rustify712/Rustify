pub fn fib4(n: usize) -> i32 {
    if n == 0 { return 0; }
    if n == 1 { return 0; }
    if n == 2 { return 2; }
    if n == 3 { return 0; }
    
    let mut f = vec![0; n + 1];
    f[0] = 0;
    f[1] = 0;
    f[2] = 2;
    f[3] = 0;
    
    for i in 4..=n {
        f[i] = f[i-1] + f[i-2] + f[i-3] + f[i-4];
    }
    
    f[n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fib4() {
        assert_eq!(fib4(0), 0);
        assert_eq!(fib4(2), 2);
        assert_eq!(fib4(4), 2);
        assert_eq!(fib4(6), 8);
    }
}