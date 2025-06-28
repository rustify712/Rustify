use md5::{Md5, Digest};
use std::fmt::Write;

pub fn string_to_md5(text: &str) -> String {
    if text.is_empty() {
        return "None".to_string();
    }

    let mut hasher = Md5::new();
    hasher.update(text.as_bytes());
    let result = hasher.finalize();

    let mut output = String::with_capacity(32);
    for byte in result {
        write!(&mut output, "{:02x}", byte).unwrap();
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_to_md5() {
        // 测试空字符串
        assert_eq!(string_to_md5(""), "None");
        
        // 测试常规字符串
        assert_eq!(string_to_md5("hello"), "5d41402abc4b2a76b9719d911017c592");
        assert_eq!(string_to_md5("rust"), "4b9c0e787a4a9a6a8e2b0b9e5b9f0e5d");
        
        // 测试特殊字符
        assert_eq!(string_to_md5("@#$%"), "a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5a5");
    }
}