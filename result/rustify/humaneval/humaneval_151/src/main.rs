/// 计算列表中所有正奇数的平方和
///
/// # 参数
/// - `lst`: 一个包含浮点数的列表
///
/// # 返回值
/// 返回所有正奇数的平方和
fn double_the_difference(lst: &[f32]) -> i64 {
    let mut sum: i64 = 0;
    for &num in lst {
        let rounded = num.round();
        if (num - rounded).abs() < 1e-4 && num > 0.0 && rounded as i64 % 2 == 1 {
            sum += (rounded as i64) * (rounded as i64);
        }
    }
    sum
}