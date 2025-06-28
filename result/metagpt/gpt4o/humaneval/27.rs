fn flip_case(s: &str) -> String {
    let mut out = String::with_capacity(s.len());

    for c in s.chars() {
        if c.is_ascii_lowercase() {
            out.push(c.to_ascii_uppercase());
        } else if c.is_ascii_uppercase() {
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }

    out
}

fn main() {
    let input = "Hello, World!";
    let flipped = flip_case(input);
    println!("Flipped case: {}", flipped);
}