fn rolling_max(numbers: &[i32]) -> Vec<i32> {
    let mut out = Vec::with_capacity(numbers.len());
    let mut max = i32::MIN;

    for &number in numbers {
        if number > max {
            max = number;
        }
        out.push(max);
    }

    out
}

fn main() {
    let numbers = vec![1, 3, 2, 5, 4];
    let result = rolling_max(&numbers);
    println!("Rolling max: {:?}", result);
}