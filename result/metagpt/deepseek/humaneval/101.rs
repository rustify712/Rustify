pub fn words_string(s: &str) -> Vec<String> {
    s.split(|c: char| c == ' ' || c == ',')
        .filter(|word| !word.is_empty())
        .map(|word| word.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_words_string() {
        assert_eq!(words_string("hello, world"), vec!["hello", "world"]);
        assert_eq!(words_string("a,b,c"), vec!["a", "b", "c"]);
        assert_eq!(words_string("  "), vec![] as Vec<String>);
    }
}