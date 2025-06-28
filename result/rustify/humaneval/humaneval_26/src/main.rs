/// Removes duplicate elements from a vector, keeping only those that appear once.
///
/// # Arguments
/// * `numbers` - A slice of integers to process.
///
/// # Returns
/// A vector containing only the elements that appear once in the input slice.
fn remove_duplicates(numbers: &[i32]) -> Vec<i32> {
    let mut has1 = Vec::new();
    let mut has2 = Vec::new();

    for &num in numbers {
        if has2.contains(&num) {
            continue;
        }
        if has1.contains(&num) {
            has2.push(num);
        } else {
            has1.push(num);
        }
    }

    numbers.iter()
        .filter(|&&num| !has2.contains(&num))
        .cloned()
        .collect()
}