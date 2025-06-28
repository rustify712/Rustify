/// 翻转字符串中字母的大小写
///
/// # 参数
/// - `s`: 需要翻转大小写的字符串
///
/// # 返回值
/// 返回翻转大小写后的字符串
fn flip_case(s: String) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_lowercase() {
                c.to_ascii_uppercase()
            } else if c.is_ascii_uppercase() {
                c.to_ascii_lowercase()
            } else {
                c
            }
        })
        .collect()
}