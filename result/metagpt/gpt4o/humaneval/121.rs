fn solutions(lst: &[i32]) -> i32 {
    let mut sum = 0;
    for i in (0..lst.len()).step_by(2) {
        if lst[i] % 2 == 1 {
            sum += lst[i];
        }
    }
    sum
}

fn main() {
    let lst = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
    let result = solutions(&lst);
    println!("Sum of odd elements at even indices: {}", result);
}