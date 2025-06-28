/// 判断一个整数是否为质数。
///
/// # 参数
/// - `n`: 需要判断的整数。
///
/// # 返回值
/// 返回 `true` 如果 `n` 是质数，否则返回 `false`。
pub fn is_prime(n: i64) -> bool {
    if n < 2 {
        return false;
    }
    let mut i = 2;
    while i * i <= n {
        if n % i == 0 {
            return false;
        }
        i += 1;
    }
    true
}