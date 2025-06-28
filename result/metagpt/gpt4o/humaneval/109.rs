fn move_one_ball(arr: &[i32]) -> bool {
    let mut num = 0;
    let size = arr.len();
    if size == 0 {
        return true;
    }
    for i in 1..size {
        if arr[i] < arr[i - 1] {
            num += 1;
        }
    }
    if arr[size - 1] > arr[0] {
        num += 1;
    }
    num < 2
}

fn main() {
    let arr = vec![3, 4, 5, 1, 2];
    let result = move_one_ball(&arr);
    println!("Can move one ball: {}", result);
}