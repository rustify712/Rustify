pub fn anti_shuffle(s: &str) -> String {
    s.split_whitespace()
        .map(|word| {
            let mut chars: Vec<char> = word.chars().collect();
            chars.sort_unstable();
            chars.into_iter().collect::<String>()
        })
        .collect::<Vec<String>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anti_shuffle() {
        assert_eq!(anti_shuffle("hello world"), "ehllo dlorw");
        assert_eq!(anti_shuffle("abc def"), "abc def");
        assert_eq!(anti_shuffle("zyx wvu"), "xyz uvw");
    }
}