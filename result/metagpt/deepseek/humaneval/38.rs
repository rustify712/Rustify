pub fn encode_cyclic(s: &str) -> String {
    let mut result = String::new();
    let mut chars: Vec<char> = s.chars().collect();
    
    for chunk in chars.chunks_mut(3) {
        if chunk.len() == 3 {
            chunk.rotate_left(1);
        }
        result.extend(chunk);
    }
    
    result
}

pub fn decode_cyclic(s: &str) -> String {
    let mut result = String::new();
    let mut chars: Vec<char> = s.chars().collect();
    
    for chunk in chars.chunks_mut(3) {
        if chunk.len() == 3 {
            chunk.rotate_right(1);
        }
        result.extend(chunk);
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let original = "abcdefgh";
        let encoded = encode_cyclic(original);
        assert_eq!(encoded, "bcaefdhg");
        assert_eq!(decode_cyclic(&encoded), original);
        
        let original = "abc";
        let encoded = encode_cyclic(original);
        assert_eq!(encoded, "bca");
        assert_eq!(decode_cyclic(&encoded), original);
    }
}