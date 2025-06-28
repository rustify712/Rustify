fn string_sequence(n: i32) -> String {
    if n < 0 {
        return String::new();
    }

    let mut out = String::from("0");
    for i in 1..=n {
        out.push_str(&format!(" {}", i));
    }
    out
}

fn main() {
    let n = 5;
    let sequence = string_sequence(n);
    println!("String sequence: {}", sequence);
}