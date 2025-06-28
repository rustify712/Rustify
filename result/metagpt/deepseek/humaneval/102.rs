pub fn choose_num(x: i32, y: i32) -> i32 {
    if y < x {
        -1
    } else if y == x && y % 2 == 1 {
        -1
    } else if y % 2 == 1 {
        y - 1
    } else {
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_choose_num() {
        assert_eq!(choose_num(1, 2), 2);
        assert_eq!(choose_num(1, 1), -1);
        assert_eq!(choose_num(1, 3), 2);
        assert_eq!(choose_num(2, 1), -1);
    }
}