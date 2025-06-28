fn f(n: usize) -> Vec<i32> {
    let mut sum = 0;
    let mut prod = 1;
    let mut out = vec![0; n];

    for i in 1..=n {
        sum += i as i32;
        prod *= i as i32;
        if i % 2 == 0 {
            out[i - 1] = prod;
        } else {
            out[i - 1] = sum;
        }
    }

    out
}

fn main() {
    let n = 5;
    let result = f(n);
    println!("Result: {:?}", result);
}