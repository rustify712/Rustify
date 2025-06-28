pub fn get_matrix_triples(n: usize) -> i32 {
    let a: Vec<i32> = (1..=n).map(|i| (i * i - i + 1) % 3).collect();
    
    let mut sum = vec![vec![0; 3]; n + 1];
    let mut sum2 = vec![vec![0; 3]; n + 1];
    
    for i in 1..=n {
        for j in 0..3 {
            sum[i][j] = sum[i-1][j];
        }
        sum[i][a[i-1] as usize] += 1;
    }
    
    for _ in 1..3 {
        for i in 1..=n {
            for j in 0..3 {
                sum2[i][j] = sum2[i-1][j];
            }
            if i >= 1 {
                for j in 0..3 {
                    sum2[i][(a[i-1] as usize + j) % 3] += sum[i-1][j];
                }
            }
        }
        
        for i in 0..=n {
            for j in 0..3 {
                sum[i][j] = sum2[i][j];
                sum2[i][j] = 0;
            }
        }
    }
    
    sum[n][0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_matrix_triples() {
        assert_eq!(get_matrix_triples(3), 1);
        assert_eq!(get_matrix_triples(5), 3);
    }
}