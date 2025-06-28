fn is_equal_to_sum_even(n: i32) -> bool {
    n % 2 == 0 && n >= 8
}

fn main() {
    let n = 10;
    let result = is_equal_to_sum_even(n);
    println!("Is {} equal to sum even condition? {}", n, result);
}