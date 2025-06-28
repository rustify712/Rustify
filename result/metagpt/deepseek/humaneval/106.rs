pub fn f(n: usize) -> Vec<i32> {
    let mut sum = 0;
    let mut prod = 1;
    let mut out = Vec::with_capacity(n);
    
    for i in 1..=n {
        sum += i as i32;
        prod *= i as i32;
        if i % 2 == 0 {
            out.push(prod);
        } else {
            out.push(sum);
        }
    }
    
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f() {
        assert_eq!(f(3), vec![1, 2, 6]);
        assert_eq!(f(4), vec![1, 2, 6, 24]);
    }
}