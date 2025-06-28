/// 判断整数 `n` 是否为质数，如果是则返回 `x`，否则返回 `y`。
///
/// # 参数
/// - `n`: 需要判断的整数。
/// - `x`: 如果 `n` 是质数，返回的值。
/// - `y`: 如果 `n` 不是质数，返回的值。
///
/// # 返回值
/// 返回 `x` 或 `y`，取决于 `n` 是否为质数。
fn x_or_y(n: i32, x: i32, y: i32) -> i32 {
    let mut is_prime = true;
    if n < 2 {
        is_prime = false;
    }
    for i in 2..=(n as f64).sqrt() as i32 {
        if n % i == 0 {
            is_prime = false;
            break;
        }
    }
    if is_prime {
        x
    } else {
        y
    }
}