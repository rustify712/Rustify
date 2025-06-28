/// Selects words from the input string `s` where the number of non-vowel letters equals `n`.
///
/// # Arguments
/// * `s` - The input string containing words separated by spaces.
/// * `n` - The number of non-vowel letters required in each selected word.
///
/// # Returns
/// A vector of strings containing the selected words.
fn select_words(s: &str, n: usize) -> Vec<String> {
    let vowels = "aeiouAEIOU";
    let mut current = String::new();
    let mut out = Vec::new();
    let mut numc = 0;
    let s = s.to_string() + " ";

    for c in s.chars() {
        if c == ' ' {
            if numc == n {
                out.push(current.clone());
            }
            current.clear();
            numc = 0;
        } else {
            current.push(c);
            if c.is_alphabetic() && !vowels.contains(c) {
                numc += 1;
            }
        }
    }

    out
}