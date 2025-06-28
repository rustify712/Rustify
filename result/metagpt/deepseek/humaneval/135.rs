pub fn can_arrange(arr: &[i32]) -> i32 {
    arr.iter().enumerate()
        .filter(|(i, &x)| x <= *i as i32)
        .map(|(i, _)| i as i32)
        .max()
        .unwrap_or(-1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_can_arrange() {
        assert_eq!(can_arrange(&[1, 0, 2]), 1);
        assert_eq!(can_arrange(&[3, 1, 2]), -1);
        assert_eq!(can_arrange(&[0, 1, 2]), 2);
    }
}