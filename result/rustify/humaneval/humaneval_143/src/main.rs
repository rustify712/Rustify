/// Returns a string containing words from the input sentence whose lengths are prime numbers.
///
/// # Arguments
///
/// * `sentence` - A string slice containing the input sentence.
///
/// # Returns
///
/// A `String` containing the words with prime lengths, separated by spaces.
fn words_in_sentence(sentence: &str) -> String {
    /// Checks if a number is prime.
    fn is_prime(n: usize) -> bool {
        if n < 2 {
            return false;
        }
        for i in 2..=(n as f64).sqrt() as usize {
            if n % i == 0 {
                return false;
            }
        }
        true
    }

    sentence
        .split_whitespace()
        .filter(|word| is_prime(word.len()))
        .collect::<Vec<&str>>()
        .join(" ")
}