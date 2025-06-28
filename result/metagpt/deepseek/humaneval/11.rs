pub fn string_xor(a: &str, b: &str) -> String {
    let max_len = a.len().max(b.len());
    let mut output = String::with_capacity(max_len);
    
    for i in 0..max_len {
        let char_a = a.chars().nth(i).unwrap_or('0');
        let char_b = b.chars().nth(i).unwrap_or('0');
        
        output.push(if char_a == char_b { '0' } else { '1' });
    }
    
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_xor() {
        assert_eq!(string_xor("1010", "1100"), "0110");
        assert_eq!(string_xor("1", "000"), "100");
    }
}