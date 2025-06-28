use std::collections::HashMap;

/// Sorts a string of space-separated number words (e.g., "one two three") and returns a sorted string.
/// The function counts the occurrences of each number word, sorts them by their numeric value,
/// and then reconstructs the string in ascending order.
///
/// # Arguments
/// * `numbers` - A string containing space-separated number words.
///
/// # Returns
/// A string with the sorted number words.
fn sort_numbers(numbers: &str) -> String {
    let tonum: HashMap<&str, i32> = [
        ("zero", 0), ("one", 1), ("two", 2), ("three", 3), ("four", 4),
        ("five", 5), ("six", 6), ("seven", 7), ("eight", 8), ("nine", 9)
    ].iter().cloned().collect();

    let numto: HashMap<i32, &str> = [
        (0, "zero"), (1, "one"), (2, "two"), (3, "three"), (4, "four"),
        (5, "five"), (6, "six"), (7, "seven"), (8, "eight"), (9, "nine")
    ].iter().cloned().collect();

    let mut count = vec![0; 10];
    for word in numbers.split_whitespace() {
        if let Some(&num) = tonum.get(word) {
            count[num as usize] += 1;
        }
    }

    let mut out = String::new();
    for i in 0..10 {
        for _ in 0..count[i] {
            out.push_str(numto[&(i as i32)]);
            out.push(' ');
        }
    }

    if !out.is_empty() {
        out.pop();
    }

    out
}
