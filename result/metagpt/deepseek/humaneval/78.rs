pub fn hex_key(num: &str) -> usize {
    const KEY: &str = "2357BD";
    num.chars().filter(|c| KEY.contains(*c)).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_key() {
        assert_eq!(hex_key("AB2357"), 4);
        assert_eq!(hex_key("ABCDEF"), 1);
        assert_eq!(hex_key("123456"), 3);
    }
}