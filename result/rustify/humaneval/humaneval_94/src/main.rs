/// 找到向量中最大的质数，并计算其各位数字之和。
///
/// # 参数
/// - `lst`: 一个包含整数的向量。
///
/// # 返回值
/// 返回最大质数的各位数字之和。
fn skjkasdkd(lst: Vec<i32>) -> i32 {
    let mut largest = 0;
    for &num in &lst {
        if num > largest && is_prime(num) {
            largest = num;
        }
    }
    largest.to_string().chars().map(|c| c.to_digit(10).unwrap() as i32).sum()
}

/// 判断一个数是否为质数。
///
/// # 参数
/// - `n`: 需要判断的整数。
///
/// # 返回值
/// 如果 `n` 是质数，返回 `true`，否则返回 `false`。
fn is_prime(n: i32) -> bool {
    if n < 2 {
        return false;
    }
    for i in 2..=(n as f64).sqrt() as i32 {
        if n % i == 0 {
            return false;
        }
    }
    true
}