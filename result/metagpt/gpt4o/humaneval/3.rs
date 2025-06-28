fn below_zero(operations: &[i32]) -> bool {
    let mut num = 0;
    for &operation in operations {
        num += operation;
        if num < 0 {
            return true;
        }
    }
    false
}

fn main() {
    let operations = vec![10, -5, -6, 3];
    let result = below_zero(&operations);
    println!("Below zero: {}", result);
}