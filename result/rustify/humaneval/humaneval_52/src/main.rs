/// 检查向量中的所有元素是否都小于给定的阈值。
///
/// # 参数
/// - `list`: 一个整数切片，表示要检查的向量。
/// - `threshold`: 一个整数，表示阈值。
///
/// # 返回值
/// 如果向量中的所有元素都小于阈值，则返回 `true`，否则返回 `false`。
fn below_threshold(list: &[i32], threshold: i32) -> bool {
    list.iter().all(|&x| x < threshold)
}