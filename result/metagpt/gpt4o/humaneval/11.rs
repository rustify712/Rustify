fn string_xor(a: &str, b: &str) -> String {
    let len_a = a.len();
    let len_b = b.len();
    let max_len = len_a.max(len_b);

    let mut output = String::with_capacity(max_len);

    for i in 0..max_len {
        let char_a = if i < len_a { a.chars().nth(i).unwrap() } else { '0' };
        let char_b = if i < len_b { b.chars().nth(i).unwrap() } else { '0' };

        if char_a == char_b {
            output.push('0');
        } else {
            output.push('1');
        }
    }

    output
}

fn main() {
    let a = "1101";
    let b = "1011";
    let result = string_xor(a, b);
    println!("XOR result: {}", result);
}