pub fn pluck(arr: &[i32]) -> Option<(i32, usize)> {
    arr.iter()
        .enumerate()
        .filter(|(_, &x)| x % 2 == 0)
        .min_by_key(|(_, &x)| x)
        .map(|(i, &x)| (x, i))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pluck() {
        assert_eq!(pluck(&[1, 2, 3, 4]), Some((2, 1)));
        assert_eq!(pluck(&[1, 3, 5]), None);
        assert_eq!(pluck(&[4, 2, 6]), Some((2, 1)));
    }
}