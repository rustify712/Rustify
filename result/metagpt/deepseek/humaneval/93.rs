pub fn encode(message: &str) -> String {
    let vowels = "aeiouAEIOU";
    message.chars().map(|c| {
        let mut c = if c.is_lowercase() {
            c.to_ascii_uppercase()
        } else if c.is_uppercase() {
            c.to_ascii_lowercase()
        } else {
            c
        };
        
        if vowels.contains(c) {
            c = ((c as u8) + 2) as char;
        }
        
        c
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode() {
        assert_eq!(encode("hello"), "JGNMQ");
        assert_eq!(encode("WORLD"), "wqtnf");
    }
}