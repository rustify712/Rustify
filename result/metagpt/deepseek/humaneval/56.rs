pub fn correct_bracketing(brackets: &str) -> bool {
    let mut level = 0;
    for c in brackets.chars() {
        match c {
            '<' => level += 1,
            '>' => {
                level -= 1;
                if level < 0 {
                    return false;
                }
            },
            _ => ()
        }
    }
    level == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correct_bracketing() {
        assert!(correct_bracketing("<<>>"));
        assert!(!correct_bracketing("<><>"));
        assert!(!correct_bracketing("<<>"));
    }
}