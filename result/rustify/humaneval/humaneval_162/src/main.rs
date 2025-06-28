use md5;

/// 计算给定字符串的 MD5 哈希值，并返回其十六进制表示。
/// 如果输入字符串为空，则返回 `None`。
///
/// # 参数
/// - `text`: 要计算 MD5 哈希值的字符串。
///
/// # 返回值
/// - `Option<String>`: 如果输入字符串为空，返回 `None`；否则返回包含 32 字符十六进制字符串的 `Some`。
pub fn string_to_md5(text: &str) -> Option<String> {
    if text.is_empty() {
        return None;
    }

    let mut context = md5::Context::new();
    context.consume(text.as_bytes());
    let result = context.compute();

    let hex_string = result.iter()
        .map(|byte| format!("{:02x}", byte))
        .collect::<String>();

    Some(hex_string)
}