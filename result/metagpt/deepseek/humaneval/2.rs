pub fn truncate_number(number: f32) -> f32 {
    number - number.floor()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_number() {
        assert_eq!(truncate_number(3.14), 0.14);
        assert_eq!(truncate_number(-1.78), 0.22);
    }
}