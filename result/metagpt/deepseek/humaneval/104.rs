pub fn unique_digits(numbers: &[i32]) -> Vec<i32> {
    let mut result: Vec<i32> = numbers.iter()
        .filter(|&&num| {
            let mut n = num;
            while n > 0 {
                if n % 2 == 0 {
                    return false;
                }
                n /= 10;
            }
            true
        })
        .cloned()
        .collect();
    
    result.sort_unstable();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unique_digits() {
        assert_eq!(unique_digits(&[13, 24, 35, 57]), vec![13, 35, 57]);
        assert_eq!(unique_digits(&[0, 1, 2, 3]), vec![1, 3]);
    }
}