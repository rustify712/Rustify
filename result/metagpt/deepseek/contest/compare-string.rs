// Rust实现字符串比较函数
// 从C版本转换而来

use std::cmp::Ordering;

// 比较两个字符串是否相等
pub fn string_equal(string1: &str, string2: &str) -> bool {
    string1 == string2
}

// 比较两个字符串的大小
pub fn string_compare(string1: &str, string2: &str) -> i32 {
    match string1.cmp(string2) {
        Ordering::Less => -1,
        Ordering::Greater => 1,
        Ordering::Equal => 0,
    }
}

// 比较两个字符串是否相等(忽略大小写)
pub fn string_nocase_equal(string1: &str, string2: &str) -> bool {
    string_nocase_compare(string1, string2) == 0
}

// 比较两个字符串的大小(忽略大小写)
pub fn string_nocase_compare(string1: &str, string2: &str) -> i32 {
    let s1_lower = string1.to_lowercase();
    let s2_lower = string2.to_lowercase();
    
    match s1_lower.cmp(&s2_lower) {
        Ordering::Less => -1,
        Ordering::Greater => 1,
        Ordering::Equal => 0,
    }
}

// 为Rust的Ord trait提供适配器
pub fn string_compare_ord(string1: &str, string2: &str) -> Ordering {
    string1.cmp(string2)
}

// 为Rust的Ord trait提供适配器(忽略大小写)
pub fn string_nocase_compare_ord(string1: &str, string2: &str) -> Ordering {
    let s1_lower = string1.to_lowercase();
    let s2_lower = string2.to_lowercase();
    s1_lower.cmp(&s2_lower)
}