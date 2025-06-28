/// 生成一个从 n 开始，每次增加 2 的数列，直到数列的长度达到 n。
///
/// # 参数
/// - `n`: 数列的起始值和长度。
///
/// # 返回值
/// 返回一个 `Vec<i32>`，包含生成的数列。
fn make_a_pile(n: i32) -> Vec<i32> {
    let mut out = vec![n];
    for i in 1..n {
        out.push(out[out.len() - 1] + 2);
    }
    out
}