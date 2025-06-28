pub fn select_words(s: &str, n: usize) -> Vec<String> {
    const VOWELS: &str = "aeiouAEIOU";
    
    s.split_whitespace()
        .filter(|word| {
            word.chars()
                .filter(|c| c.is_ascii_alphabetic() && !VOWELS.contains(*c))
                .count() == n
        })
        .map(|word| word.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_select_words() {
        assert_eq!(select_words("Hello world", 3), vec!["Hello"]);
        assert_eq!(select_words("Rust is awesome", 2), vec!["Rust", "awesome"]);
        assert!(select_words("No consonants", 0).is_empty());
    }
}