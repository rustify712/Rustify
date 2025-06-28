fn largest_divisor(n: i32) -> i32 {
    for i in 2..=((n as f64).sqrt() as i32) {
        if n % i == 0 {
            return n / i;
        }
    }
    1
}

fn main() {
    let number = 100;
    let divisor = largest_divisor(number);
    println!("Largest divisor of {} is {}", number, divisor);
}