/// 解析音乐符号字符串并返回对应的整数值向量。
///
/// # 参数
/// - `music_string`: 包含音乐符号的字符串，符号之间用空格分隔。
///
/// # 返回值
/// 返回一个包含解析后整数值的向量。
fn parse_music(music_string: &str) -> Vec<i32> {
    let mut current = String::new();
    let mut out = Vec::new();
    let music_string = if !music_string.is_empty() {
        format!("{} ", music_string)
    } else {
        music_string.to_string()
    };

    for c in music_string.chars() {
        if c == ' ' {
            match current.as_str() {
                "o" => out.push(4),
                "o|" => out.push(2),
                ".|" => out.push(1),
                _ => {},
            }
            current.clear();
        } else {
            current.push(c);
        }
    }
    out
}