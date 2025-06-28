fn get_matrix_triples(n: usize) -> i32 {
    let mut a = vec![0; n];
    let mut sum = vec![vec![0; 3]; n + 1];
    let mut sum2 = vec![vec![0; 3]; n + 1];

    for i in 1..=n {
        a[i - 1] = (i * i - i + 1) % 3;
        for j in 0..3 {
            sum[i][j] = sum[i - 1][j];
        }
        sum[i][a[i - 1]] += 1;
    }

    for _ in 1..3 {
        for i in 1..=n {
            for j in 0..3 {
                sum2[i][j] = sum2[i - 1][j];
            }
            if i >= 1 {
                for j in 0..3 {
                    sum2[i][(a[i - 1] + j) % 3] += sum[i - 1][j];
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

fn main() {
    let n = 5;
    let result = get_matrix_triples(n);
    println!("Result: {}", result);
}