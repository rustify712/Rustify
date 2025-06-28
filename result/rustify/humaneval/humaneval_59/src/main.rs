/// 计算给定整数 `n` 的最大质因数。
///
/// # 参数
/// - `n`: 需要计算最大质因数的整数。
///
/// # 返回值
/// 返回 `n` 的最大质因数。
fn largest_prime_factor(mut n: i32) -> i32 {
    let mut i = 2;
    while i * i <= n {
        while n % i == 0 && n > i {
            n /= i;
        }
        i += 1;
    }
    n
}