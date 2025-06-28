fn rounded_avg(n: i32, m: i32) -> String {
    if n > m {
        return String::from("-1");
    }

    let mut num = (m + n) / 2;
    let mut out = String::new();

    if num == 0 {
        return String::from("0");
    }

    while num > 0 {
        out.insert(0, char::from_digit((num % 2) as u32, 10).unwrap());
        num /= 2;
    }

    out
}

fn main() {
    let n = 5;
    let m = 10;
    let result = rounded_avg(n, m);
    println!("Rounded average in binary: {}", result);
}