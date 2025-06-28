pub fn count_nums(nums: &[i32]) -> i32 {
    nums.iter().filter(|&&n| {
        if n > 0 {
            true
        } else {
            let mut sum = 0;
            let mut w = n.abs();
            while w >= 10 {
                sum += w % 10;
                w /= 10;
            }
            sum -= w;
            sum > 0
        }
    }).count() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_count_nums() {
        assert_eq!(count_nums(&[1, -22, 3, -45]), 3);
        assert_eq!(count_nums(&[0, -1, -10]), 0);
    }
}