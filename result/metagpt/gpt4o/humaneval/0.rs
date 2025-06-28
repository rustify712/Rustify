fn has_close_elements(numbers: &[f32], threshold: f32) -> bool {
    for i in 0..numbers.len() {
        for j in (i + 1)..numbers.len() {
            if (numbers[i] - numbers[j]).abs() < threshold {
                return true;
            }
        }
    }
    false
}

fn main() {
    let numbers = vec![1.0, 2.0, 3.0, 4.0];
    let threshold = 0.5;
    let result = has_close_elements(&numbers, threshold);
    println!("Has close elements: {}", result);
}