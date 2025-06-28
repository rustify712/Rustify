/// 计算两个整数的最大公约数（GCD）。
///
/// # 参数
/// - `a`: 第一个整数。
/// - `b`: 第二个整数。
///
/// # 返回值
/// 返回 `a` 和 `b` 的最大公约数。
fn greatest_common_divisor(mut a: i32, mut b: i32) -> i32 {
    loop {
        if a < b {
            (a, b) = (b, a);
        }
        a = a % b;
        if a == 0 {
            return b;
        }
    }
}