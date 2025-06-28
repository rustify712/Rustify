pub fn x_or_y(n: i32, x: i32, y: i32) -> i32 {
    if n < 2 {
        return y;
    }
    
    for i in 2..=(n as f64).sqrt() as i32 {
        if n % i == 0 {
            return y;
        }
    }
    
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_x_or_y() {
        assert_eq!(x_or_y(7, 10, 20), 10); // 7是质数
        assert_eq!(x_or_y(8, 10, 20), 20); // 8不是质数
        assert_eq!(x_or_y(1, 10, 20), 20); // 1不是质数
    }
}