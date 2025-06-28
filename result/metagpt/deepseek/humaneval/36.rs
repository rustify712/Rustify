pub fn fizz_buzz(n: i32) -> i32 {
    (0..n)
        .filter(|&i| i % 11 == 0 || i % 13 == 0)
        .map(|mut q| {
            let mut count = 0;
            while q > 0 {
                if q % 10 == 7 {
                    count += 1;
                }
                q /= 10;
            }
            count
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fizz_buzz() {
        assert_eq!(fizz_buzz(20), 2);
        assert_eq!(fizz_buzz(50), 5);
    }
}