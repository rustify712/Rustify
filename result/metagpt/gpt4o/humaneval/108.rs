fn count_nums(n: &[i32]) -> i32 {
    let mut num = 0;
    for &value in n.iter() {
        if value > 0 {
            num += 1;
        } else {
            let mut sum = 0;
            let mut w = value.abs();
            while w >= 10 {
                sum += w % 10;
                w /= 10;
            }
            sum -= w;
            if sum > 0 {
                num += 1;
            }
        }
    }
    num
}

fn main() {
    let numbers = vec![12, -34, 56, -78, 90];
    let result = count_nums(&numbers);
    println!("Count of numbers meeting criteria: {}", result);
}