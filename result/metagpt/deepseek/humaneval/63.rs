pub fn fibfib(n: usize) -> i32 {
    if n == 0 { return 0; }
    if n == 1 { return 0; }
    if n == 2 { return 1; }
    
    let mut ff = vec![0; n + 1];
    ff[0] = 0;
    ff[1] = 0;
    ff[2] = 1;
    
    for i in 3..=n {
        ff[i] = ff[i-1] + ff[i-2] + ff[i-3];
    }
    
    ff[n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibfib() {
        assert_eq!(fibfib(0), 0);
        assert_eq!(fibfib(1), 0);
        assert_eq!(fibfib(2), 1);
        assert_eq!(fibfib(5), 4);
        assert_eq!(fibfib(8), 24);
    }
}