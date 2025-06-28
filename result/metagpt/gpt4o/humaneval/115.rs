fn max_fill(grid: &[Vec<i32>], capacity: i32) -> i32 {
    let mut out = 0;
    for row in grid.iter() {
        let sum: i32 = row.iter().sum();
        if sum > 0 {
            out += (sum - 1) / capacity + 1;
        }
    }
    out
}

fn main() {
    let grid = vec![
        vec![1, 2, 3],
        vec![4, 5, 6],
        vec![7, 8, 9],
    ];
    let capacity = 5;
    let result = max_fill(&grid, capacity);
    println!("Maximum fill: {}", result);
}