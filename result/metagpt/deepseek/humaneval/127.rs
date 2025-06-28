pub fn intersection(interval1: (i32, i32), interval2: (i32, i32)) -> &'static str {
    let start = interval1.0.max(interval2.0);
    let end = interval1.1.min(interval2.1);
    let length = end - start;
    
    if length < 2 {
        return "NO";
    }
    
    for i in 2..=(length as f64).sqrt() as i32 {
        if length % i == 0 {
            return "NO";
        }
    }
    
    "YES"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intersection() {
        assert_eq!(intersection((1, 5), (3, 7)), "YES"); // 交集长度3是质数
        assert_eq!(intersection((1, 3), (5, 7)), "NO"); // 无交集
        assert_eq!(intersection((2, 6), (3, 5)), "NO"); // 交集长度2不是质数
    }
}