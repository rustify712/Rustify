use std::collections::HashMap;

#[derive(Debug)]
struct CharCount {
    key: char,
    value: usize,
}

fn histogram(test: &str) -> Vec<CharCount> {
    let mut count = HashMap::new();
    let mut max = 0;

    // Count the frequency of each character
    for ch in test.chars() {
        if ch != ' ' {
            let counter = count.entry(ch).or_insert(0);
            *counter += 1;
            if *counter > max {
                max = *counter;
            }
        }
    }

    // Collect characters with the maximum frequency
    let mut result = Vec::new();
    for (&key, &value) in count.iter() {
        if value == max {
            result.push(CharCount { key, value });
        }
    }

    result
}

fn main() {
    let test_str = "example string with some characters";
    let result = histogram(test_str);
    println!("Characters with maximum frequency: {:?}", result);
}