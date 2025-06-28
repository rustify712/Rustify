pub fn get_closest_vowel(word: &str) -> String {
    const VOWELS: &str = "AEIOUaeiou";
    
    for (i, c) in word.chars().enumerate().rev().skip(1).take(word.len() - 2) {
        if VOWELS.contains(c) {
            let prev = word.chars().nth(i - 1).unwrap();
            let next = word.chars().nth(i + 1).unwrap();
            if !VOWELS.contains(prev) && !VOWELS.contains(next) {
                return c.to_string();
            }
        }
    }
    
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_closest_vowel() {
        assert_eq!(get_closest_vowel("apple"), "");
        assert_eq!(get_closest_vowel("banana"), "a");
        assert_eq!(get_closest_vowel("hello"), "");
    }
}