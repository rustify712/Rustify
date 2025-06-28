pub fn generate_integers(a: i32, b: i32) -> Vec<i32> {
    let (start, end) = if a < b { (a, b) } else { (b, a) };
    
    (start..=end)
        .filter(|&x| x < 10 && x % 2 == 0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_integers() {
        // 测试正常范围
        assert_eq!(generate_integers(2, 8), vec![2, 4, 6, 8]);
        assert_eq!(generate_integers(8, 2), vec![2, 4, 6, 8]);
        
        // 测试包含大于10的数
        assert_eq!(generate_integers(5, 15), vec![6, 8]);
        assert_eq!(generate_integers(15, 5), vec![6, 8]);
        
        // 测试无结果情况
        assert_eq!(generate_integers(10, 20), vec![]);
        assert_eq!(generate_integers(1, 1), vec![]);
        
        // 测试边界情况
        assert_eq!(generate_integers(8, 8), vec![8]);
        assert_eq!(generate_integers(9, 9), vec![]);
    }
}