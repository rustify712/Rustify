fn is_consonant(c: char) -> bool {
    !matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'A' | 'E' | 'I' | 'O' | 'U')
}

fn select_words(s: &str, n: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut numc = 0;

    for ch in s.chars().chain(Some(' ')) {
        if ch == ' ' || ch == '\0' {
            if numc == n {
                out.push(current.clone());
            }
            current.clear();
            numc = 0;
        } else {
            current.push(ch);
            if ch.is_alphabetic() && is_consonant(ch) {
                numc += 1;
            }
        }
    }

    out
}

fn main() {
    let s = "hello world this is a test";
    let n = 3;
    let result = select_words(s, n);
    for word in result {
        println!("{}", word);
    }
}