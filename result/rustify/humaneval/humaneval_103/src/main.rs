/// 计算两个整数的平均值并将其转换为二进制字符串。
///
/// # 参数
/// - `n`: 第一个整数。
/// - `m`: 第二个整数。
///
/// # 返回值
/// - 如果 `n` 大于 `m`，返回 `None`。
/// - 否则，返回 `Some(String)`，其中包含平均值的二进制表示。
fn rounded_avg(n: i32, m: i32) -> Option<String> {
    if n > m {
        return None;
    }
    let num = (n + m) / 2;
    Some(format!("{:b}", num))
}