pub fn split_words(txt: &str) -> Vec<String> {
    // 检查是否包含空格或逗号
    if txt.contains(' ') || txt.contains(',') {
        return txt.split(|c| c == ' ' || c == ',')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
    }
    
    // 统计小写字母且字母顺序为奇数的数量
    let count = txt.chars()
        .filter(|c| c.is_ascii_lowercase())
        .filter(|c| (*c as u8 - b'a') % 2 == 0)
        .count();
    
    vec![count.to_string()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_words() {
        assert_eq!(split_words("hello world"), vec!["hello", "world"]);
        assert_eq!(split_words("a,b,c"), vec!["a", "b", "c"]);
        assert_eq!(split_words("abc"), vec!["1"]); // a(0), c(2) 符合条件
    }
}