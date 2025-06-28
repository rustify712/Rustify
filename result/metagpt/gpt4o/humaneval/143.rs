fn words_in_sentence(sentence: &str) -> String {
    let mut out = String::new();
    let mut current = String::new();

    for ch in sentence.chars().chain(Some(' ')) {
        if ch != ' ' && ch != '\0' {
            current.push(ch);
        } else {
            let l = current.len();
            let mut isp = true;
            if l < 2 {
                isp = false;
            } else {
                for j in 2..=((l as f64).sqrt() as usize) {
                    if l % j == 0 {
                        isp = false;
                        break;
                    }
                }
            }
            if isp {
                out.push_str(&current);
                out.push(' ');
            }
            current.clear();
        }
    }

    // Remove the trailing space if any
    if !out.is_empty() {
        out.pop();
    }

    out
}

fn main() {
    let sentence = "This is a test sentence with some words";
    let result = words_in_sentence(sentence);
    println!("Words with prime length: {}", result);
}