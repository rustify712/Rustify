/// 返回一个包含所有大于 0 的元素的新 Vec<f32>。
///
/// # 参数
/// - `list`: 输入的浮点数列表。
///
/// # 返回值
/// 返回一个新的 Vec<f32>，其中包含所有大于 0 的元素。
fn get_positive(list: &[f32]) -> Vec<f32> {
    list.iter().filter(|&&x| x > 0.0).cloned().collect()
}