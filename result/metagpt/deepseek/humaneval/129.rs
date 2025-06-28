pub fn min_path(grid: Vec<Vec<i32>>, k: usize) -> Vec<i32> {
    // 查找值为1的位置
    let (x, y) = grid.iter().enumerate()
        .find_map(|(i, row)| {
            row.iter().position(|&val| val == 1)
                .map(|j| (i, j))
        }).unwrap();
    
    // 查找四个方向的最小值
    let mut min = i32::MAX;
    if x > 0 && grid[x-1][y] < min { min = grid[x-1][y]; }
    if x < grid.len() - 1 && grid[x+1][y] < min { min = grid[x+1][y]; }
    if y > 0 && grid[x][y-1] < min { min = grid[x][y-1]; }
    if y < grid[0].len() - 1 && grid[x][y+1] < min { min = grid[x][y+1]; }
    
    // 生成结果数组
    (0..k).map(|i| if i % 2 == 0 { 1 } else { min }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_path() {
        let grid = vec![vec![2, 1, 3], vec![4, 5, 6]];
        assert_eq!(min_path(grid, 3), vec![1, 2, 1]);
    }
}