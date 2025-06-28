use std::collections::HashMap;

/// Searches for the maximum value in the list that satisfies the condition.
///
/// # Arguments
///
/// * `lst` - A slice of integers to search through.
///
/// # Returns
///
/// The maximum value that satisfies the condition, or -1 if no such value exists.
fn search(lst: &[i32]) -> i32 {
    let mut freq = HashMap::new();
    let mut max = -1;

    for &num in lst {
        let count = freq.entry(num).or_insert(0);
        *count += 1;
        if *count >= num && num > max {
            max = num;
        }
    }

    max
}
