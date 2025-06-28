/// 计算满足条件的整数个数
///
/// # 参数
/// - `n`: 一个包含整数的切片
///
/// # 返回值
/// 返回满足条件的整数个数
fn count_nums(n: &[i32]) -> i32 {
    let mut num = 0;
    for &item in n {
        if item > 0 {
            num += 1;
        } else {
            let mut sum = 0;
            let mut w = item.abs();
            while w >= 10 {
                sum += w % 10;
                w /= 10;
            }
            sum -= w;
            if sum > 0 {
                num += 1;
            }
        }
    }
    num
}