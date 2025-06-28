fn sum_squares(lst: &[i32]) -> i32 {
    let mut sum = 0;
    for (i, &value) in lst.iter().enumerate() {
        if i % 3 == 0 {
            sum += value * value;
        } else if i % 4 == 0 {
            sum += value * value * value;
        } else {
            sum += value;
        }
    }
    sum
}

fn main() {
    let lst = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
    let result = sum_squares(&lst);
    println!("Sum of squares: {}", result);
}