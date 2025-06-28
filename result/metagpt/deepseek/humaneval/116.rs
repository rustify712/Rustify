pub fn sort_array(mut arr: Vec<i32>) -> Vec<i32> {
    // 计算每个数字的二进制1的个数
    let mut counts: Vec<(i32, u32)> = arr.iter()
        .map(|&x| (x, x.abs().count_ones()))
        .collect();
    
    // 根据1的个数和原始值排序
    counts.sort_by(|a, b| {
        a.1.cmp(&b.1)
            .then_with(|| a.0.cmp(&b.0))
    });
    
    // 提取排序后的数组
    counts.into_iter()
        .map(|(x, _)| x)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_array() {
        assert_eq!(sort_array(vec![1, 2, 3, 4]), vec![1, 2, 4, 3]);
        assert_eq!(sort_array(vec![-1, -2, -3]), vec![-2, -1, -3]);
    }
}