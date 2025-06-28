fn is_palindrome(s: &str) -> bool {
    let len = s.len();
    for i in 0..len / 2 {
        if s.chars().nth(i) != s.chars().nth(len - 1 - i) {
            return false;
        }
    }
    true
}

fn make_palindrome(s: &str) -> String {
    let len = s.len();
    for i in 0..len {
        if is_palindrome(&s[i..]) {
            let mut result = String::from(s);
            let prefix: String = s[..i].chars().rev().collect();
            result.push_str(&prefix);
            return result;
        }
    }
    let mut result = String::from(s);
    let reverse: String = s.chars().rev().collect();
    result.push_str(&reverse);
    result
}

fn main() {
    let test_cases = vec!["", "cat", "cata"];
    for &test in &test_cases {
        let palindrome = make_palindrome(test);
        println!("Original: {}, Palindrome: {}", test, palindrome);
    }
}