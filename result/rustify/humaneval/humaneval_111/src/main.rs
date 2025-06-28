use std::collections::HashMap;

/// 统计字符串中每个字符的出现次数，并返回出现次数最多的字符及其出现次数。
///
/// # 参数
/// - `test`: 输入的字符串。
///
/// # 返回值
/// 返回一个 `HashMap<char, i32>`，包含出现次数最多的字符及其出现次数。
fn histogram(test: &str) -> HashMap<char, i32> {
    let mut count = HashMap::new();
    let mut max = 0;

    // 统计每个字符的出现次数
    for c in test.chars() {
        if c != ' ' {
            let counter = count.entry(c).or_insert(0);
            *counter += 1;
            if *counter > max {
                max = *counter;
            }
        }
    }

    // 找出出现次数最多的字符
    count.into_iter().filter(|&(_, v)| v == max).collect()
}