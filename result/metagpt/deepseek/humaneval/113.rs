pub fn odd_count(lst: &[&str]) -> Vec<String> {
    lst.iter().map(|s| {
        let count = s.chars()
            .filter(|c| c.is_ascii_digit())
            .filter(|c| c.to_digit(10).unwrap() % 2 == 1)
            .count();
        format!("the number of odd elements in the string {} of the input.", count)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_odd_count() {
        assert_eq!(
            odd_count(&["123", "456"]),
            vec![
                "the number of odd elements in the string 2 of the input.",
                "the number of odd elements in the string 0 of the input."
            ]
        );
    }
}