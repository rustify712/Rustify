pub fn greatest_common_divisor(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greatest_common_divisor() {
        assert_eq!(greatest_common_divisor(3, 5), 1);
        assert_eq!(greatest_common_divisor(25, 15), 5);
    }
}