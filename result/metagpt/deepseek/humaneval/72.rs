pub fn will_it_fly(q: &[i32], w: i32) -> bool {
    // 检查是否对称
    if !q.iter().eq(q.iter().rev()) {
        return false;
    }
    
    // 检查总和是否超过阈值
    q.iter().sum::<i32>() <= w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_will_it_fly() {
        assert!(will_it_fly(&[1, 2, 1], 5));
        assert!(!will_it_fly(&[1, 2, 3], 5));
        assert!(!will_it_fly(&[1, 2, 2, 1], 4));
    }
}