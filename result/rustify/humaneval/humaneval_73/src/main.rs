/// 计算将向量变为回文向量所需的最小修改次数。
///
/// # 参数
/// - `arr`: 一个整数向量，表示需要检查的向量。
///
/// # 返回值
/// 返回一个整数，表示需要修改的最小次数。
fn smallest_change(arr: &Vec<i32>) -> i32 {
    let mut out = 0;
    for i in 0..arr.len() / 2 {
        if arr[i] != arr[arr.len() - 1 - i] {
            out += 1;
        }
    }
    out
}