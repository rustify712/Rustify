fn fix_spaces(text: &str) -> String {
    let len = text.len();
    let mut out = String::with_capacity(2 * len); // Allocate enough space for the output
    let mut spacelen = 0;

    for ch in text.chars() {
        if ch == ' ' {
            spacelen += 1;
        } else {
            if spacelen == 1 {
                out.push('_');
            } else if spacelen == 2 {
                out.push('_');
                out.push('_');
            } else if spacelen > 2 {
                out.push('-');
            }
            spacelen = 0;
            out.push(ch);
        }
    }

    // Handle trailing spaces
    if spacelen == 1 {
        out.push('_');
    } else if spacelen == 2 {
        out.push('_');
        out.push('_');
    } else if spacelen > 2 {
        out.push('-');
    }

    out
}

fn main() {
    let text = "This  is   a test    string";
    let result = fix_spaces(text);
    println!("Fixed spaces: {}", result);
}