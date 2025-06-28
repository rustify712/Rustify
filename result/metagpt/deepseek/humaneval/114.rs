pub fn min_sub_array_sum(nums: &[i64]) -> i64 {
    if nums.is_empty() {
        return 0;
    }
    
    let mut current = nums[0];
    let mut min = nums[0];
    
    for &num in &nums[1..] {
        if current < 0 {
            current += num;
        } else {
            current = num;
        }
        
        if current < min {
            min = current;
        }
    }
    
    min
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_sub_array_sum() {
        assert_eq!(min_sub_array_sum(&[1, 2, 3]), 1);
        assert_eq!(min_sub_array_sum(&[-1, -2, -3]), -6);
        assert_eq!(min_sub_array_sum(&[3, -1, 2, -1]), -1);
    }
}