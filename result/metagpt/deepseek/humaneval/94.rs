pub fn skjkasdkd(lst: &[i32]) -> i32 {
    // 找出最大质数
    let largest_prime = lst.iter()
        .filter(|&&x| x > 1)
        .filter(|&&x| (2..=(x as f64).sqrt() as i32).all(|i| x % i != 0))
        .max()
        .copied()
        .unwrap_or(0);
    
    // 计算各位数字之和
    largest_prime.to_string().chars()
        .map(|c| c.to_digit(10).unwrap() as i32)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skjkasdkd() {
        assert_eq!(skjkasdkd(&[2, 4, 7, 11]), 2); // 11 -> 1+1=2
        assert_eq!(skjkasdkd(&[4, 6, 8]), 0); // 无质数
    }
}