/// 计算字符串中所有大写字母的 ASCII 码之和
///
/// # 参数
/// - `s`: 输入的字符串
///
/// # 返回值
/// 返回字符串中所有大写字母的 ASCII 码之和
fn digit_sum(s: &str) -> u32 {
    let mut sum = 0;
    for c in s.chars() {
        if c.is_ascii_uppercase() {
            sum += c as u32;
        }
    }
    sum
}