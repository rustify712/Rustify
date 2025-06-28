pub fn encode_shift(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_lowercase() {
                ((((c as u8 - b'a') + 5) % 26) + b'a') as char
            } else {
                c
            }
        })
        .collect()
}

pub fn decode_shift(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_lowercase() {
                ((((c as u8 - b'a') + 21) % 26) + b'a') as char
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
    fn test_encode_decode() {
        let original = "hello";
        let encoded = encode_shift(original);
        assert_eq!(encoded, "mjqqt");
        assert_eq!(decode_shift(&encoded), original);
    }
}