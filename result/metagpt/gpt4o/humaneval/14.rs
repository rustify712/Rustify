fn all_prefixes(s: &str) -> Vec<String> {
    let len = s.len();
    let mut out = Vec::with_capacity(len);

    for i in 0..len {
        let prefix = &s[..=i];
        out.push(prefix.to_string());
    }

    out
}

fn main() {
    let input = "hello";
    let prefixes = all_prefixes(input);
    for prefix in prefixes {
        println!("{}", prefix);
    }
}