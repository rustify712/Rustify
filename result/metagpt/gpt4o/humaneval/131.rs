fn digits(n: i32) -> i32 {
    let mut prod = 1;
    let mut has_odd_digit = false;
    let s = n.to_string();

    for ch in s.chars() {
        let digit = ch.to_digit(10).unwrap();
        if digit % 2 == 1 {
            has_odd_digit = true;
            prod *= digit as i32;
        }
    }

    if !has_odd_digit {
        return 0;
    }
    prod
}

fn main() {
    let n = 123456;
    let result = digits(n);
    println!("Product of odd digits: {}", result);
}