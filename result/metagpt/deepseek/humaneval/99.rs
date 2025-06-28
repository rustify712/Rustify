pub fn closest_integer(value: &str) -> i32 {
    value.parse::<f64>().unwrap().round() as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_closest_integer() {
        assert_eq!(closest_integer("10.4"), 10);
        assert_eq!(closest_integer("10.5"), 11);
        assert_eq!(closest_integer("-10.4"), -10);
    }
}