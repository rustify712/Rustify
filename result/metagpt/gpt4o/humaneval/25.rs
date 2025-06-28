fn factorize(n: i32) -> Vec<i32> {
    let mut factors = Vec::new();
    let mut num = n;

    for i in 2..=((num as f64).sqrt() as i32) {
        while num % i == 0 {
            factors.push(i);
            num /= i;
        }
    }

    if num > 1 {
        factors.push(num);
    }

    factors
}

fn main() {
    let number = 100;
    let factors = factorize(number);
    println!("Factors of {} are {:?}", number, factors);
}