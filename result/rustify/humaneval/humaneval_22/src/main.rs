/// 定义一个类型别名 `ListAny`，表示一个可以存储任意类型元素的列表。
/// 使用 `Vec<Box<dyn Any>>` 来替代 C++ 中的 `std::list<boost::any>`。
type ListAny = Vec<Box<dyn std::any::Any>>;

/// Filters integers from a vector of boxed `Any` values.
///
/// # Arguments
/// * `values` - A vector of boxed `Any` values.
///
/// # Returns
/// A vector of integers extracted from the input values.
fn filter_integers(values: Vec<Box<dyn std::any::Any>>) -> Vec<i32> {
    let mut out = Vec::new();
    for value in values {
        if let Some(int_value) = value.downcast_ref::<i32>() {
            out.push(*int_value);
        }
    }
    out
}