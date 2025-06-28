fn sort(arr: &mut [i32]) {
    let size = arr.len();
    for i in 0..size - 1 {
        for j in 0..size - i - 1 {
            if arr[j] > arr[j + 1] {
                arr.swap(j, j + 1);
            }
        }
    }
}

fn reverse(arr: &mut [i32]) {
    let size = arr.len();
    for i in 0..size / 2 {
        arr.swap(i, size - i - 1);
    }
}

fn num_to_word(num: i32) -> Option<&'static str> {
    match num {
        0 => Some("Zero"),
        1 => Some("One"),
        2 => Some("Two"),
        3 => Some("Three"),
        4 => Some("Four"),
        5 => Some("Five"),
        6 => Some("Six"),
        7 => Some("Seven"),
        8 => Some("Eight"),
        9 => Some("Nine"),
        _ => None,
    }
}

fn by_length(arr: &[i32]) -> Vec<String> {
    let mut arr = arr.to_vec();
    sort(&mut arr);
    reverse(&mut arr);

    let mut result = Vec::new();
    for &num in arr.iter() {
        if let Some(word) = num_to_word(num) {
            result.push(word.to_string());
        }
    }
    result
}

fn main() {
    let numbers = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
    let result = by_length(&numbers);
    println!("Sorted and reversed numbers as words: {:?}", result);
}