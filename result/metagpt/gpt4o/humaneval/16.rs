use std::collections::HashSet;

fn count_distinct_characters(s: &str) -> usize {
    let mut distinct = HashSet::new();

    for c in s.chars() {
        distinct.insert(c.to_ascii_lowercase());
    }

    distinct.len()
}

fn main() {
    let input = "Hello, World!";
    let count = count_distinct_characters(input);
    println!("Number of distinct characters: {}", count);
}