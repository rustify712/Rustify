pub fn order_by_points(mut nums: Vec<i32>) -> Vec<i32> {
    #[derive(Debug)]
    struct NumWithSum {
        num: i32,
        sum: i32,
    }
    
    let mut nums_with_sum: Vec<NumWithSum> = nums.iter()
        .map(|&n| {
            let mut sum = 0;
            let mut num = n.abs();
            while num > 0 {
                sum += num % 10;
                num /= 10;
            }
            if n < 0 {
                sum = -sum;
            }
            NumWithSum { num: n, sum }
        })
        .collect();
    
    nums_with_sum.sort_by(|a, b| a.sum.cmp(&b.sum));
    
    nums_with_sum.into_iter().map(|x| x.num).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_by_points() {
        assert_eq!(order_by_points(vec![123, -456, 789]), vec![-456, 123, 789]);
        assert_eq!(order_by_points(vec![0, 1, -1]), vec![0, -1, 1]);
    }
}