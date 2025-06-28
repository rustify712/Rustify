pub fn find_max(words: &[&str]) -> &str {
    let mut max_word = "";
    let mut max_unique = 0;

    for word in words {
        let mut unique = [false; 256]; // ASCII characters
        let mut unique_count = 0;

        for &ch in word.as_bytes() {
            if !unique[ch as usize] {
                unique[ch as usize] = true;
                unique_count += 1;
            }
        }

        if unique_count > max_unique || 
           (unique_count == max_unique && word < &max_word) {
            max_word = word;
            max_unique = unique_count;
        }
    }

    max_word
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_max() {
        let words1 = ["apple", "banana", "orange", "pear"];
        assert_eq!(find_max(&words1), "orange");

        let words2 = ["hello", "world", "rust", "code"];
        assert_eq!(find_max(&words2), "world");

        let words3 = ["a", "b", "c", "d"];
        assert_eq!(find_max(&words3), "a");

        let words4 = ["abc", "aab", "abb", "aaa"];
        assert_eq!(find_max(&words4), "abc");

        let words5 = ["unique", "letters", "test"];
        assert_eq!(find_max(&words5), "letters");
    }
}