pub fn sum_product(numbers: &[i32]) -> (i32, i32) {
    let sum = numbers.iter().sum();
    let product = numbers.iter().product();
    (sum, product)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_product() {
        assert_eq!(sum_product(&[1, 2, 3, 4]), (10, 24));
        assert_eq!(sum_product(&[5, -2]), (3, -10));
    }
}