fn greatest_common_divisor(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        let m = a % b;
        a = b;
        b = m;
    }
    a
}

fn main() {
    let test_cases = vec![(3, 5), (25, 15)];
    for &(a, b) in &test_cases {
        let gcd = greatest_common_divisor(a, b);
        println!("GCD of {} and {} is {}", a, b, gcd);
    }
}