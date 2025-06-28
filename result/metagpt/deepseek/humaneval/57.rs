pub fn monotonic(l: &[f32]) -> bool {
    let mut incr = false;
    let mut decr = false;
    
    for i in 1..l.len() {
        if l[i] > l[i-1] {
            incr = true;
        } else if l[i] < l[i-1] {
            decr = true;
        }
    }
    
    !(incr && decr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monotonic() {
        assert!(monotonic(&[1.0, 2.0, 3.0]));
        assert!(monotonic(&[3.0, 2.0, 1.0]));
        assert!(!monotonic(&[1.0, 3.0, 2.0]));
    }
}