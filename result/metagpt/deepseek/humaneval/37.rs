pub fn sort_even(l: &[f32]) -> Vec<f32> {
    let mut result = l.to_vec();
    
    // 提取并排序偶数索引元素
    let mut even_indices: Vec<_> = l.iter()
        .enumerate()
        .filter(|(i, _)| i % 2 == 0)
        .map(|(_, &x)| x)
        .collect();
    
    even_indices.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    // 将排序后的元素放回原位置
    for (i, &x) in even_indices.iter().enumerate() {
        result[i * 2] = x;
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_even() {
        assert_eq!(
            sort_even(&[5.0, 3.0, 2.0, 8.0, 1.0, 4.0]),
            vec![1.0, 3.0, 2.0, 8.0, 5.0, 4.0]
        );
        assert_eq!(
            sort_even(&[4.0, 1.0, 2.0]),
            vec![2.0, 1.0, 4.0]
        );
    }
}