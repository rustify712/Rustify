pub fn solve(n: i32) -> String {
    let sum: u32 = n.to_string().chars().map(|c| c.to_digit(10).unwrap()).sum();
    format!("{:b}", sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve() {
        assert_eq!(solve(123), "110"); // 1+2+3=6, 6 in binary is 110
        assert_eq!(solve(9999), "10011000100"); // 9+9+9+9=36, 36 in binary is 100100
    }
}