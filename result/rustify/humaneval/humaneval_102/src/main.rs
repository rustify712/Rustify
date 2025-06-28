/// 根据给定的 x 和 y 返回一个整数。
///
/// # 参数
/// - `x`: 整数，表示范围的下限。
/// - `y`: 整数，表示范围的上限。
///
/// # 返回值
/// - 如果 y 小于 x 或者 y 等于 x 且 y 是奇数，返回 `None`。
/// - 如果 y 是奇数，返回 `Some(y - 1)`。
/// - 否则，返回 `Some(y)`。
fn choose_num(x: i32, y: i32) -> Option<i32> {
    if y < x {
        return None;
    }
    if y == x && y % 2 == 1 {
        return None;
    }
    if y % 2 == 1 {
        return Some(y - 1);
    }
    Some(y)
}