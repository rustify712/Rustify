// Rust实现指针比较函数
// 从C版本转换而来

use std::cmp::Ordering;

// 比较两个指针是否相等
pub fn pointer_equal<T>(location1: *const T, location2: *const T) -> bool {
    location1 == location2
}

// 比较两个指针的大小
pub fn pointer_compare<T>(location1: *const T, location2: *const T) -> i32 {
    if (location1 as usize) < (location2 as usize) {
        -1
    } else if (location1 as usize) > (location2 as usize) {
        1
    } else {
        0
    }
}

// 为Rust的Ord trait提供适配器
pub fn pointer_compare_ord<T>(location1: *const T, location2: *const T) -> Ordering {
    if (location1 as usize) < (location2 as usize) {
        Ordering::Less
    } else if (location1 as usize) > (location2 as usize) {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}