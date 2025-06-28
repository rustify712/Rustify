/// 计算给定整数数组中的最小子数组和。
///
/// # 参数
/// - `nums`: 一个包含整数的数组。
///
/// # 返回值
/// 返回数组中的最小子数组和。
fn min_sub_array_sum(nums: &[i64]) -> i64 {
    let mut current = nums[0];
    let mut min = nums[0];
    for &num in &nums[1..] {
        if current < 0 {
            current += num;
        } else {
            current = num;
        }
        if current < min {
            min = current;
        }
    }
    min
}