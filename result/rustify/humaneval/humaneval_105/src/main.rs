use std::collections::HashMap;

/// Converts a vector of integers to a vector of their corresponding English words.
/// Only integers between 1 and 9 (inclusive) are converted, and the result is sorted in descending order.
///
/// # Arguments
///
/// * `arr` - A vector of integers to be converted.
///
/// # Returns
///
/// A vector of strings representing the English words of the integers.
fn by_length(arr: Vec<i32>) -> Vec<&'static str> {
    let numto: HashMap<i32, &'static str> = [
        (0, "Zero"),
        (1, "One"),
        (2, "Two"),
        (3, "Three"),
        (4, "Four"),
        (5, "Five"),
        (6, "Six"),
        (7, "Seven"),
        (8, "Eight"),
        (9, "Nine"),
    ]
    .iter()
    .cloned()
    .collect();

    let mut sorted_arr = arr;
    sorted_arr.sort();

    let mut out = Vec::new();
    for &num in sorted_arr.iter().rev() {
        if num >= 1 && num <= 9 {
            out.push(numto[&num]);
        }
    }

    out
}