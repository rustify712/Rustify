use std::collections::HashMap;

pub fn histogram(s: &str) -> Vec<(char, usize)> {
    let mut counts = HashMap::new();
    
    // 统计字符频率(忽略空格)
    for c in s.chars().filter(|&c| c != ' ') {
        *counts.entry(c).or_insert(0) += 1;
    }
    
    if counts.is_empty() {
        return vec![];
    }
    
    // 找出最大频率
    let max_count = *counts.values().max().unwrap();
    
    // 收集所有达到最大频率的字符
    counts.into_iter()
        .filter(|&(_, count)| count == max_count)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram() {
        let result = histogram("hello world");
        assert!(result.contains(&('l', 3)));
        
        let empty = histogram("   ");
        assert!(empty.is_empty());
    }
}