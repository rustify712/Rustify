pub fn rounded_avg(n: i32, m: i32) -> String {
    if n > m {
        return "-1".to_string();
    }
    
    let avg = (n + m) / 2;
    if avg == 0 {
        return "0".to_string();
    }
    
    format!("{:b}", avg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rounded_avg() {
        assert_eq!(rounded_avg(1, 5), "10"); // (1+5)/2=3, 3 in binary is 11
        assert_eq!(rounded_avg(5, 1), "-1");
        assert_eq!(rounded_avg(0, 0), "0");
    }
}