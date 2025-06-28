fn intersperse(numbers: &[i32], delimiter: i32) -> Vec<i32> {
    if numbers.is_empty() {
        return Vec::new();
    }

    let mut out_size = 2 * numbers.len() - 1;
    let mut out = Vec::with_capacity(out_size);

    out.push(numbers[0]);
    for &number in &numbers[1..] {
        out.push(delimiter);
        out.push(number);
    }

    out
}

fn main() {
    let numbers = vec![1, 2, 3, 4];
    let delimiter = 0;
    let result = intersperse(&numbers, delimiter);
    println!("Interspersed array: {:?}", result);
}