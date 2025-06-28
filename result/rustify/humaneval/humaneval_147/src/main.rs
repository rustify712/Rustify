/// 计算与矩阵相关的三元组的值
///
/// # 参数
/// - `n`: 矩阵的大小
///
/// # 返回值
/// 返回计算得到的值
fn get_matrix_triples(n: usize) -> i32 {
    let mut a = Vec::new();
    let mut sum = vec![vec![0, 0, 0]];
    let mut sum2 = vec![vec![0, 0, 0]];

    for i in 1..=n {
        a.push((i * i - i + 1) % 3);
        sum.push(sum[sum.len() - 1].clone());
        sum[i][a[i - 1] as usize] += 1;
    }

    for _ in 1..3 {
        for i in 1..=n {
            sum2.push(sum2[sum2.len() - 1].clone());
            if i >= 1 {
                for j in 0..=2 {
                    sum2[i][(a[i - 1] + j) % 3 as usize] += sum[i - 1][j as usize];
                }
            }
        }
        sum = sum2.clone();
        sum2 = vec![vec![0, 0, 0]];
    }

    sum[n][0]
}