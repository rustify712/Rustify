pub fn parse_music(music_string: &str) -> Vec<i32> {
    let mut result = Vec::new();
    let mut current = String::new();
    
    for c in music_string.chars() {
        if c == ' ' {
            match current.as_str() {
                "o" => result.push(4),
                "o|" => result.push(2),
                ".|" => result.push(1),
                _ => {}
            }
            current.clear();
        } else {
            current.push(c);
        }
    }
    
    // 处理最后一个音符
    match current.as_str() {
        "o" => result.push(4),
        "o|" => result.push(2),
        ".|" => result.push(1),
        _ => {}
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_music() {
        assert_eq!(parse_music("o o| .|"), vec![4, 2, 1]);
        assert_eq!(parse_music("o o o|"), vec![4, 4, 2]);
    }
}