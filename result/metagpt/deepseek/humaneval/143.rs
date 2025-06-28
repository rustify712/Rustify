pub fn words_in_sentence(sentence: &str) -> String {
    fn is_prime(n: usize) -> bool {
        if n < 2 {
            return false;
        }
        for i in 2..=(n as f64).sqrt() as usize {
            if n % i == 0 {
                return false;
            }
        }
        true
    }
    
    sentence.split_whitespace()
        .filter(|word| is_prime(word.len()))
        .collect::<Vec<&str>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_words_in_sentence() {
        assert_eq!(words_in_sentence("This is a test"), "is a");
        assert_eq!(words_in_sentence("Rust is awesome"), "is");
        assert_eq!(words_in_sentence("Hello world"), "");
    }
}