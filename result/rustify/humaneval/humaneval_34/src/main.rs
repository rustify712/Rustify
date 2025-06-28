/// 返回一个去重并排序后的整数向量。
///
/// # 参数
/// - `l`: 输入的整数向量。
///
/// # 返回值
/// 返回一个去重并排序后的新向量。
fn unique(l: &[i32]) -> Vec<i32> {
    use std::collections::HashSet;
    let mut set: HashSet<i32> = HashSet::new();
    for &item in l {
        set.insert(item);
    }
    let mut out: Vec<i32> = set.into_iter().collect();
    out.sort();
    out
}