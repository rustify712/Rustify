pub fn starts_one_ends(n: u32) -> u32 {
    if n < 1 {
        return 0;
    }
    if n == 1 {
        return 1;
    }
    let mut out = 18;
    for _ in 2..n {
        out *= 10;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_starts_one_ends() {
        assert_eq!(starts_one_ends(1), 1);
        assert_eq!(starts_one_ends(2), 18);
        assert_eq!(starts_one_ends(3), 180);
    }
}