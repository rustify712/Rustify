/// 对整数进行循环移位操作
///
/// # 参数
/// - `x`: 需要进行循环移位的整数
/// - `shift`: 移位的位数
///
/// # 返回值
/// 返回循环移位后的字符串
fn circular_shift(x: i32, shift: usize) -> String {
    let xs = x.to_string();
    if xs.len() < shift {
        return xs.chars().rev().collect();
    }
    let split_point = xs.len() - shift;
    let (left, right) = xs.split_at(split_point);
    format!("{}{}", right, left)
}