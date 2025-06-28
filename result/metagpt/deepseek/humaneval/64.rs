pub fn vowels_count(s: &str) -> usize {
    let vowels: Vec<char> = "aeiouAEIOU".chars().collect();
    let mut count = s.chars().filter(|c| vowels.contains(c)).count();
    
    if let Some(last_char) = s.chars().last() {
        if last_char == 'y' || last_char == 'Y' {
            count += 1;
        }
    }
    
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vowels_count() {
        assert_eq!(vowels_count("hello"), 2);
        assert_eq!(vowels_count("why"), 2);
        assert_eq!(vowels_count("aeiou"), 5);
        assert_eq!(vowels_count("AEIOUY"), 6);
        assert_eq!(vowels_count(""), 0);
    }
}