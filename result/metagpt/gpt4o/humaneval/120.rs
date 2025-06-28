fn compare(a: &i32, b: &i32) -> std::cmp::Ordering {
    a.cmp(b)
}

fn maximum(arr: &mut [i32], k: usize) -> Vec<i32> {
    // Sort the array in ascending order
    arr.sort_by(compare);

    // Collect the last k elements from the sorted array
    arr[arr.len() - k..].to_vec()
}

fn main() {
    let mut arr = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
    let k = 3;
    let result = maximum(&mut arr, k);
    println!("Maximum {} numbers: {:?}", k, result);
}