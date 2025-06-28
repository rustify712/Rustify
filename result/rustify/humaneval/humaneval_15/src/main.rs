/// 生成一个从 "0" 开始的字符串，依次追加从 1 到 n 的数字，每个数字之间用空格分隔。
///
/// # 参数
/// - `n`: 整数，表示要追加的最大数字。
///
/// # 返回值
/// 返回一个字符串，包含从 "0" 到 `n` 的数字，每个数字之间用空格分隔。
fn string_sequence(n: i32) -> String {
    let mut out = String::from("0");
    for i in 1..=n {
        out.push_str(&format!(" {}", i));
    }
    out
}