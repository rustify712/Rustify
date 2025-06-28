/// 对字符串中的每个单词按字母顺序排序，并返回排序后的字符串。
///
/// # 参数
/// - `s`: 输入的字符串。
///
/// # 返回值
/// 返回一个 `String`，其中每个单词的字符已按字母顺序排序。
fn anti_shuffle(s: &str) -> String {
    let mut out = String::new();
    for word in s.split_whitespace() {
        let mut chars: Vec<char> = word.chars().collect();
        chars.sort();
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&chars.into_iter().collect::<String>());
    }
    out
}