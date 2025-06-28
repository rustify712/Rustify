fn add_elements(arr: &[i32], k: usize) -> i32 {
    let mut sum = 0;
    for i in 0..k.min(arr.len()) {
        if arr[i] >= -99 && arr[i] <= 99 {
            sum += arr[i];
        }
    }
    sum
}

fn main() {
    let arr = vec![100, -50, 30, 99, -100, 50];
    let k = 4;
    let result = add_elements(&arr, k);
    println!("Sum of elements within range: {}", result);
}