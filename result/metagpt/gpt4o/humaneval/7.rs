fn filter_by_substring(strings: &[&str], substring: &str) -> Vec<&str> {
    let mut out = Vec::new();

    for &string in strings {
        if string.contains(substring) {
            out.push(string);
        }
    }

    out
}

fn main() {
    let strings = vec!["hello", "world", "hell", "hello world"];
    let substring = "hell";
    let filtered_strings = filter_by_substring(&strings, substring);
    println!("Filtered strings: {:?}", filtered_strings);
}