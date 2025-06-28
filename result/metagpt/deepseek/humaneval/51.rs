pub fn remove_vowels(text: &str) -> String {
    text.chars()
        .filter(|c| !"AEIOUaeiou".contains(*c))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_vowels() {
        assert_eq!(remove_vowels("abcdef"), "bcdf");
        assert_eq!(remove_vowels("Hello World"), "Hll Wrld");
        assert_eq!(remove_vowels(""), "");
    }
}