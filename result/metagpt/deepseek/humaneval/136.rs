pub fn largest_smallest_integers(lst: &[i32]) -> (Option<i32>, Option<i32>) {
    let maxneg = lst.iter().filter(|&&x| x < 0).max().copied();
    let minpos = lst.iter().filter(|&&x| x > 0).min().copied();
    
    (maxneg, minpos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_largest_smallest_integers() {
        assert_eq!(largest_smallest_integers(&[1, -2, 3]), (Some(-2), Some(1)));
        assert_eq!(largest_smallest_integers(&[-1, -2, -3]), (Some(-1), None));
        assert_eq!(largest_smallest_integers(&[1, 2, 3]), (None, Some(1)));
    }
}