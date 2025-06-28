pub fn prod_signs(arr: &[i32]) -> i32 {
    if arr.is_empty() {
        return -32768;
    }
    
    let sum: i32 = arr.iter().map(|x| x.abs()).sum();
    let sign_prod = arr.iter().fold(1, |acc, &x| {
        if x == 0 {
            0
        } else if x < 0 {
            -acc
        } else {
            acc
        }
    });
    
    sum * sign_prod
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prod_signs() {
        assert_eq!(prod_signs(&[1, -2, 3]), -6); // 1+2+3=6, 1*-1*1=-1 => 6*-1=-6
        assert_eq!(prod_signs(&[]), -32768);
        assert_eq!(prod_signs(&[0, 1, 2]), 0);
    }
}