fn odd_count(lst: &[&str]) -> Vec<String> {
    let mut out = Vec::new();
    for &item in lst.iter() {
        let mut sum = 0;
        for ch in item.chars() {
            if ch.is_digit(10) && ch.to_digit(10).unwrap() % 2 == 1 {
                sum += 1;
            }
        }
        let s = "the number of odd elements in the string i of the input.";
        let mut s2 = String::new();
        for ch in s.chars() {
            if ch == 'i' {
                s2.push_str(&sum.to_string());
            } else {
                s2.push(ch);
            }
        }
        out.push(s2);
    }
    out
}

fn main() {
    let lst = vec!["123", "456", "789"];
    let result = odd_count(&lst);
    for res in result {
        println!("{}", res);
    }
}