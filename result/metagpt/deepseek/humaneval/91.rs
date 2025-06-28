pub fn is_bored(s: &str) -> usize {
    let mut count = 0;
    let mut is_start = true;
    
    for c in s.chars() {
        if is_start && c == 'I' {
            count += 1;
            is_start = false;
        } else if c == '.' || c == '?' || c == '!' {
            is_start = true;
        } else if c != ' ' {
            is_start = false;
        }
    }
    
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_bored() {
        assert_eq!(is_bored("I am bored. I want to sleep."), 2);
        assert_eq!(is_bored("Hello world!"), 0);
        assert_eq!(is_bored("I think, therefore I am."), 1);
    }
}