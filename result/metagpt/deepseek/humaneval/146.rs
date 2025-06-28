pub fn special_filter(nums: &[i32]) -> usize {
    nums.iter()
        .filter(|&&x| x > 10)
        .filter(|&&x| {
            let s = x.abs().to_string();
            let first = s.chars().next().unwrap().to_digit(10).unwrap();
            let last = s.chars().last().unwrap().to_digit(10).unwrap();
            first % 2 == 1 && last % 2 == 1
        })
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_filter() {
        assert_eq!(special_filter(&[11, 12, 13, 14]), 2); // 11和13符合
        assert_eq!(special_filter(&[21, 23, 25]), 3);
        assert_eq!(special_filter(&[10, 20, 30]), 0);
    }
}