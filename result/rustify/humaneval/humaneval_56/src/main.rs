/// 检查字符串中的括号是否正确匹配。
///
/// # 参数
/// - `brackets`: 包含括号的字符串。
///
/// # 返回值
/// - 返回 `true` 如果括号正确匹配，否则返回 `false`。
fn correct_bracketing(brackets: &str) -> bool {
    let mut level = 0;
    for c in brackets.chars() {
        match c {
            '<' => level += 1,
            '>' => level -= 1,
            _ => {}
        }
        if level < 0 {
            return false;
        }
    }
    level == 0
}