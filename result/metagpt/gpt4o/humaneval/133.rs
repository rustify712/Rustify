fn sum_squares(lst: &[f32]) -> i32 {
    let mut sum = 0;
    for &value in lst.iter() {
        let ceil_value = value.ceil() as i32;
        sum += ceil_value * ceil_value;
    }
    sum
}

fn main() {
    let lst = vec![1.2, 2.5, 3.7];
    let result = sum_squares(&lst);
    println!("Sum of squares: {}", result);
}