pub fn get_odd_collatz(n: u32) -> Vec<u32> {
    let mut result = vec![1];
    let mut current = n;
    
    while current != 1 {
        if current % 2 == 1 {
            result.push(current);
            current = current * 3 + 1;
        } else {
            current /= 2;
        }
    }
    
    result.sort_unstable();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_odd_collatz() {
        assert_eq!(get_odd_collatz(5), vec![1, 5]);
        assert_eq!(get_odd_collatz(1), vec![1]);
        assert_eq!(get_odd_collatz(6), vec![1]);
    }
}