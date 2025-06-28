fn char_in_string(ch: char, s: &str) -> bool {
    s.contains(ch)
}

fn reverse_string(s: &mut String) {
    let len = s.len();
    for i in 0..len / 2 {
        s.as_mut_vec().swap(i, len - i - 1);
    }
}

fn reverse_delete(s: &str, c: &str) -> (String, bool) {
    let mut n: String = s.chars().filter(|&ch| !char_in_string(ch, c)).collect();
    
    if n.is_empty() {
        return (n, true);
    }

    let mut w = n.clone();
    reverse_string(&mut w);

    let is_palindrome = w == n;
    (n, is_palindrome)
}

fn main() {
    let s = "example";
    let c = "ae";
    let (result, is_palindrome) = reverse_delete(s, c);
    println!("Resulting string: {}", result);
    println!("Is palindrome: {}", is_palindrome);
}