/// 计算满足条件的整数数量
///
/// # 参数
/// - `nums`: 一个包含整数的向量
///
/// # 返回值
/// 返回满足条件的整数数量
fn special_filter(nums: Vec<i32>) -> i32 {
    let mut num = 0;
    for &n in nums.iter() {
        if n > 10 {
            let w = n.to_string();
            let first_char = w.chars().next().unwrap();
            let last_char = w.chars().last().unwrap();
            if first_char.to_digit(10).unwrap() % 2 == 1 && last_char.to_digit(10).unwrap() % 2 == 1 {
                num += 1;
            }
        }
    }
    num
}