pub fn simplify(x: &str, n: &str) -> bool {
    fn parse_fraction(s: &str) -> Option<(i32, i32)> {
        let parts: Vec<&str> = s.split('/').collect();
        if parts.len() != 2 {
            return None;
        }
        let numerator = parts[0].parse::<i32>().ok()?;
        let denominator = parts[1].parse::<i32>().ok()?;
        Some((numerator, denominator))
    }
    
    let (a, b) = match parse_fraction(x) {
        Some((a, b)) => (a, b),
        None => return false,
    };
    
    let (c, d) = match parse_fraction(n) {
        Some((c, d)) => (c, d),
        None => return false,
    };
    
    (a * c) % (b * d) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify() {
        assert!(simplify("1/2", "2/1"));
        assert!(!simplify("1/3", "2/3"));
        assert!(!simplify("1/2", "invalid"));
    }
}