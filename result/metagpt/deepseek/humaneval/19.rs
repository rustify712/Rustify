use std::collections::HashMap;

pub fn sort_numbers(numbers: &str) -> String {
    let num_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
    let mut word_to_num = HashMap::new();
    
    for (i, &word) in num_words.iter().enumerate() {
        word_to_num.insert(word, i);
    }
    
    let mut counts = [0; 10];
    
    for word in numbers.split_whitespace() {
        if let Some(&num) = word_to_num.get(word) {
            counts[num] += 1;
        }
    }
    
    let mut result = Vec::new();
    
    for (i, &count) in counts.iter().enumerate() {
        for _ in 0..count {
            result.push(num_words[i]);
        }
    }
    
    result.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_numbers() {
        assert_eq!(
            sort_numbers("three one two one three"),
            "one one two three three"
        );
        assert_eq!(
            sort_numbers("zero nine eight seven"),
            "zero seven eight nine"
        );
    }
}