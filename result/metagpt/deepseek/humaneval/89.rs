pub fn encrypt(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_lowercase() {
                ((((c as u8 - b'a') + 4) % 26) + b'a') as char
            } else {
                c
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt() {
        assert_eq!(encrypt("abc"), "efg");
        assert_eq!(encrypt("xyz"), "bcd");
        assert_eq!(encrypt("hello"), "lipps");
    }
}