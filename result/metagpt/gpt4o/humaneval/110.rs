fn exchange(lst1: &[i32], lst2: &[i32]) -> &'static str {
    let mut num = 0;
    for &item in lst1.iter() {
        if item % 2 == 0 {
            num += 1;
        }
    }
    for &item in lst2.iter() {
        if item % 2 == 0 {
            num += 1;
        }
    }
    if num >= lst1.len() {
        "YES"
    } else {
        "NO"
    }
}

fn main() {
    let lst1 = vec![1, 2, 3, 4];
    let lst2 = vec![5, 6, 7, 8];
    let result = exchange(&lst1, &lst2);
    println!("Exchange result: {}", result);
}