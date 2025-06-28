pub fn even_odd_count(num: i32) -> (i32, i32) {
    let num_str = num.abs().to_string();
    let mut even_count = 0;
    let mut odd_count = 0;

    for ch in num_str.chars() {
        if let Some(digit) = ch.to_digit(10) {
            if digit % 2 == 0 {
                even_count += 1;
            } else {
                odd_count += 1;
            }
        }
    }

    (even_count, odd_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_even_odd_count() {
        assert_eq!(even_odd_count(123456), (3, 3));
        assert_eq!(even_odd_count(13579), (0, 5));
        assert_eq!(even_odd_count(24680), (5, 0));
        assert_eq!(even_odd_count(-12345), (2, 3));
        assert_eq!(even_odd_count(0), (1, 0));
    }
}