fn check_if_last_char_is_a_letter(txt: &str) -> bool {
    let length = txt.len();
    if length == 0 {
        return false;
    }

    let chr = txt.chars().nth(length - 1).unwrap();
    if !chr.is_alphabetic() {
        return false;
    }

    if length == 1 {
        return true;
    }

    let prev_chr = txt.chars().nth(length - 2).unwrap();
    if prev_chr.is_alphabetic() {
        return false;
    }

    true
}

fn main() {
    let txt = "Hello!";
    let result = check_if_last_char_is_a_letter(txt);
    println!("Is the last character a letter? {}", result);
}