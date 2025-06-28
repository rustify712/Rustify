pub fn tri(n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n + 1);
    
    out.push(1);
    if n == 0 {
        return out;
    }
    
    out.push(3);
    for i in 2..=n {
        if i % 2 == 0 {
            out.push(1 + (i / 2) as i32);
        } else {
            let val = out[i-1] + out[i-2] + 1 + ((i + 1) / 2) as i32;
            out.push(val);
        }
    }
    
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tri() {
        assert_eq!(tri(0), vec![1]);
        assert_eq!(tri(1), vec![1, 3]);
        assert_eq!(tri(3), vec![1, 3, 2, 8]);
    }
}