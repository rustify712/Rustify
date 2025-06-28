fn get_odd_collatz(mut n: i32) -> Vec<i32> {
    let mut out = Vec::new();

    // Add 1 to the result array
    out.push(1);

    while n != 1 {
        if n % 2 == 1 {
            out.push(n);
            n = n * 3 + 1;
        } else {
            n /= 2;
        }
    }

    // Sort the array
    out.sort();

    out
}

fn main() {
    let n = 5;
    let result = get_odd_collatz(n);

    for num in result.iter() {
        print!("{} ", num);
    }
}