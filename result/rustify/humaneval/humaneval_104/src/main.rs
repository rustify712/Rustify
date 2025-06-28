/// 返回一个包含所有数字都是奇数的整数的向量，并且返回的向量是排序后的。
///
/// # 参数
/// - `x`: 输入的整数切片
///
/// # 返回值
/// 返回一个包含所有数字都是奇数的整数的向量，并且是排序后的。
fn unique_digits(x: &[i32]) -> Vec<i32> {
    let mut out = Vec::new();
    for &num in x {
        let mut u = true;
        let mut n = num;
        if n == 0 {
            u = false;
        }
        while n > 0 && u {
            if n % 2 == 0 {
                u = false;
            }
            n /= 10;
        }
        if u {
            out.push(num);
        }
    }
    out.sort();
    out
}