fn longest(strings: &[&str]) -> Option<&str> {
    if strings.is_empty() {
        return None;
    }

    let mut out = strings[0];
    for &string in &strings[1..] {
        if string.len() > out.len() {
            out = string;
        }
    }
    Some(out)
}

fn main() {
    let strings = vec!["short", "longer", "longest"];
    match longest(&strings) {
        Some(longest_str) => println!("Longest string: {}", longest_str),
        None => println!("No strings provided."),
    }
}