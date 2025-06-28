/// 计算以1开头或以1结尾的n位数的数量。
///
/// # 参数
/// - `n`: 数字的位数。
///
/// # 返回值
/// 返回以1开头或以1结尾的n位数的数量。
fn starts_one_ends(n: i32) -> i32 {
    if n < 1 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    let out = 18;
    out * 10i32.pow((n - 2) as u32)
}
