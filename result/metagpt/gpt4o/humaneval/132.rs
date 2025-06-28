fn is_nested(str: &str) -> bool {
    let mut count = 0;
    let mut maxcount = 0;

    for ch in str.chars() {
        if ch == '[' {
            count += 1;
        } else if ch == ']' {
            count -= 1;
        }

        if count < 0 {
            count = 0;
        }

        if count > maxcount {
            maxcount = count;
        }

        if count <= maxcount - 2 {
            return true;
        }
    }

    false
}

fn main() {
    let test_str = "[[[]]]";
    let result = is_nested(test_str);
    println!("Is the string nested? {}", result);
}