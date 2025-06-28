pub fn pairs_sum_to_zero(numbers: &[i32]) -> bool {
    for i in 0..numbers.len() {
        for j in i+1..numbers.len() {
            if numbers[i] + numbers[j] == 0 {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pairs_sum_to_zero() {
        assert!(pairs_sum_to_zero(&[1, -1, 2]));
        assert!(!pairs_sum_to_zero(&[1, 2, 3]));
    }
}