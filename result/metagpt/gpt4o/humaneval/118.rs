fn get_closest_vowel(word: &str) -> String {
    let vowels = "AEIOUaeiou";
    let len = word.len();
    let chars: Vec<char> = word.chars().collect();

    for i in (1..len - 1).rev() {
        if vowels.contains(chars[i]) {
            if !vowels.contains(chars[i + 1]) && !vowels.contains(chars[i - 1]) {
                return chars[i].to_string();
            }
        }
    }

    String::new() // Return an empty string if no suitable vowel is found
}

fn main() {
    let word = "example";
    let closest_vowel = get_closest_vowel(word);
    println!("Closest vowel: {}", closest_vowel);
}