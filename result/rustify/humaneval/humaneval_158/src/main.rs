/// Finds the string with the maximum number of unique characters.
/// If multiple strings have the same number of unique characters, returns the lexicographically smallest one.
///
/// # Arguments
/// * `words` - A vector of strings to search through.
///
/// # Returns
/// The string with the most unique characters, or the lexicographically smallest one if there is a tie.
fn find_max(words: &Vec<String>) -> String {
    let mut max = String::new();
    let mut maxu = 0;

    for word in words {
        let unique: std::collections::HashSet<char> = word.chars().collect();
        let unique_len = unique.len();

        if unique_len > maxu || (unique_len == maxu && word < &max) {
            max = word.clone();
            maxu = unique_len;
        }
    }

    max
}