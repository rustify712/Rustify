fn split_words(txt: &str) -> Vec<String> {
    let mut out = Vec::new();

    // Check if the text contains whitespace
    if txt.contains(' ') {
        let temp = format!("{} ", txt);
        let mut current = String::new();
        for ch in temp.chars() {
            if ch == ' ' {
                if !current.is_empty() {
                    out.push(current.clone());
                }
                current.clear();
            } else {
                current.push(ch);
            }
        }
        return out;
    }

    // Check if the text contains commas
    if txt.contains(',') {
        let temp = format!("{},", txt);
        let mut current = String::new();
        for ch in temp.chars() {
            if ch == ',' {
                if !current.is_empty() {
                    out.push(current.clone());
                }
                current.clear();
            } else {
                current.push(ch);
            }
        }
        return out;
    }

    // If no whitespace or commas, count lowercase letters with odd order
    let num = txt.chars().filter(|&ch| ch.is_ascii_lowercase() && (ch as u8 - b'a') % 2 == 0).count();
    out.push(num.to_string());

    out
}

fn main() {
    let txt = "hello,world,example";
    let result = split_words(txt);
    for word in result {
        println!("{}", word);
    }
}