fn concatenate(strings: &[&str]) -> String {
    let total_length: usize = strings.iter().map(|s| s.len()).sum();
    let mut out = String::with_capacity(total_length);

    for &string in strings {
        out.push_str(string);
    }

    out
}

fn main() {
    let strings = vec!["Hello", " ", "World", "!"];
    let result = concatenate(&strings);
    println!("Concatenated string: {}", result);
}