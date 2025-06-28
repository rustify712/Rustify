pub fn solve(s: &str) -> String {
    let mut nletter = 0;
    let mut out: Vec<char> = s.chars().collect();
    let len = out.len();

    for i in 0..len {
        let w = out[i];
        if w.is_uppercase() {
            out[i] = w.to_ascii_lowercase();
        } else if w.is_lowercase() {
            out[i] = w.to_ascii_uppercase();
        } else {
            nletter += 1;
        }
    }

    if nletter == len {
        out.reverse();
    }

    out.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        // 测试大小写转换
        assert_eq!(solve("Hello"), "hELLO");
        assert_eq!(solve("Rust"), "rUST");
        
        // 测试全非字母情况
        assert_eq!(solve("12345"), "54321");
        assert_eq!(solve("@#$%^"), "^%$#@");
        
        // 测试混合情况
        assert_eq!(solve("He1lO"), "hE1Lo");
        assert_eq!(solve("aBc123"), "AbC123");
        
        // 测试空字符串
        assert_eq!(solve(""), "");
    }
}