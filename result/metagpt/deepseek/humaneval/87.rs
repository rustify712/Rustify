pub struct Coordinate {
    pub row: usize,
    pub col: usize,
}

pub fn get_row(lst: &[Vec<i32>], x: i32) -> Vec<Coordinate> {
    let mut out = Vec::new();
    
    for (i, row) in lst.iter().enumerate() {
        for (j, &val) in row.iter().enumerate().rev() {
            if val == x {
                out.push(Coordinate { row: i, col: j });
            }
        }
    }
    
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_row() {
        let input = vec![vec![1, 2, 3], vec![4, 5, 6], vec![7, 8, 9]];
        let result = get_row(&input, 5);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].row, 1);
        assert_eq!(result[0].col, 1);
    }
}