/// 对两个字符串进行逐字符异或操作，返回结果字符串。
///
/// # 参数
/// - `a`: 第一个字符串。
/// - `b`: 第二个字符串。
///
/// # 返回值
/// 返回一个新的字符串，表示 `a` 和 `b` 的逐字符异或结果。
fn string_xor(a: &str, b: &str) -> String {
    let mut output = String::new();
    let mut a_chars = a.chars();
    let mut b_chars = b.chars();

    loop {
        match (a_chars.next(), b_chars.next()) {
            (Some(a_char), Some(b_char)) => {
                if a_char == b_char {
                    output.push('0');
                } else {
                    output.push('1');
                }
            }
            (Some(a_char), None) => output.push(a_char),
            (None, Some(b_char)) => output.push(b_char),
            (None, None) => break,
        }
    }

    output
}