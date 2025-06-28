/// 生成小于给定整数 `n` 的所有质数。
///
/// # 参数
/// - `n`: 一个整数，表示生成质数的上限。
///
/// # 返回值
/// 返回一个包含所有小于 `n` 的质数的 `Vec<i32>`。
fn count_up_to(n: i32) -> Vec<i32> {
    let mut out = Vec::new();
    for i in 2..n {
        if out.is_empty() {
            out.push(i);
        } else {
            let mut is_prime = true;
            for &prime in &out {
                if prime * prime > i {
                    break;
                }
                if i % prime == 0 {
                    is_prime = false;
                    break;
                }
            }
            if is_prime {
                out.push(i);
            }
        }
    }
    out
}