/// 根据数组的第一个元素和最后一个元素的和的奇偶性来决定如何排序数组。
/// 如果和为奇数，则按升序排序；如果和为偶数，则按降序排序。
///
/// # 参数
/// - `array`: 输入的整数数组。
///
/// # 返回值
/// 返回排序后的数组。
fn sort_array(array: &[i32]) -> Vec<i32> {
    if array.is_empty() {
        return Vec::new();
    }
    let mut sorted_array = array.to_vec();
    sorted_array.sort();
    if (array[0] + array[array.len() - 1]) % 2 == 1 {
        sorted_array
    } else {
        sorted_array.reverse();
        sorted_array
    }
}