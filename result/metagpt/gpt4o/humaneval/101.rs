fn words_string(s: &str) -> Vec<String> {
    let mut words = Vec::new();
    
    // Split string by whitespace and commas, filter out empty segments
    for word in s.split(|c| c == ' ' || c == ',').filter(|&word| !word.is_empty()) {
        words.push(word.to_string());
    }
    
    words
}

fn main() {
    let input = "Hello, world, this is a test";
    let words = words_string(input);
    println!("{:?}", words);
}