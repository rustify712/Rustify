// Rust实现整数比较函数
// 从C版本转换而来

// 比较两个整数是否相等
pub fn int_equal(location1: &i32, location2: &i32) -> bool {
    *location1 == *location2
}

// 比较两个整数的大小
pub fn int_compare(location1: &i32, location2: &i32) -> i32 {
    if *location1 < *location2 {
        -1
    } else if *location1 > *location2 {
        1
    } else {
        0
    }
}

// 为Rust的Ord trait提供适配器
pub fn int_compare_ord(location1: &i32, location2: &i32) -> std::cmp::Ordering {
    if *location1 < *location2 {
        std::cmp::Ordering::Less
    } else if *location1 > *location2 {
        std::cmp::Ordering::Greater
    } else {
        std::cmp::Ordering::Equal
    }
}