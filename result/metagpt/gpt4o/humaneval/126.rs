fn is_sorted(lst: &[i32]) -> bool {
    for i in 1..lst.len() {
        if lst[i] < lst[i - 1] {
            return false;
        }
        if i >= 2 && lst[i] == lst[i - 1] && lst[i] == lst[i - 2] {
            return false;
        }
    }
    true
}

fn main() {
    let lst = vec![1, 2, 2, 3, 4];
    let result = is_sorted(&lst);
    println!("Is the list sorted? {}", result);
}