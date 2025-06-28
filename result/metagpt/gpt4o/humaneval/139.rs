fn special_factorial(n: i32) -> i64 {
    let mut fact: i64 = 1;
    let mut bfact: i64 = 1;
    for i in 1..=n {
        fact *= i as i64;
        bfact *= fact;
    }
    bfact
}

fn main() {
    let n = 5;
    let result = special_factorial(n);
    println!("Special factorial of {}: {}", n, result);
}