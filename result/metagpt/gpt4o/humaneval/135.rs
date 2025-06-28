fn can_arrange(arr: &[i32]) -> i32 {
    let mut max = -1;
    for (i, &value) in arr.iter().enumerate() {
        if value <= i as i32 {
            max = i as i32;
        }
    }
    max
}

fn main() {
    let arr = vec![0, 2, 1, 3, 5];
    let result = can_arrange(&arr);
    println!("Maximum index where arrangement is possible: {}", result);
}