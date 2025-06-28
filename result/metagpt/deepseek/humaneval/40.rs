pub fn triples_sum_to_zero(numbers: &[i32]) -> bool {
    for i in 0..numbers.len() {
        for j in i+1..numbers.len() {
            for k in j+1..numbers.len() {
                if numbers[i] + numbers[j] + numbers[k] == 0 {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triples_sum_to_zero() {
        assert!(triples_sum_to_zero(&[1, -2, 1]));
        assert!(!triples_sum_to_zero(&[1, 2, 3]));
    }
}