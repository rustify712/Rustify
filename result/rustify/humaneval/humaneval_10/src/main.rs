/// Checks if a given string is a palindrome.
///
/// # Arguments
/// * `s` - A string slice to check.
///
/// # Returns
/// * `bool` - `true` if the string is a palindrome, `false` otherwise.
fn is_palindrome(s: &str) -> bool {
    s.chars().rev().collect::<String>() == s
}

/// Finds the shortest palindrome that begins with a supplied string.
///
/// # Arguments
/// * `s` - A string slice to make into a palindrome.
///
/// # Returns
/// * `String` - The shortest palindrome that begins with the supplied string.
fn make_palindrome(s: &str) -> String {
    for i in 0..s.len() {
        let rstr = &s[i..];
        if is_palindrome(rstr) {
            let nstr = &s[..i];
            let n2str = nstr.chars().rev().collect::<String>();
            return format!("{}{}", s, n2str);
        }
    }
    let n2str = s.chars().rev().collect::<String>();
    format!("{}{}", s, n2str)
}