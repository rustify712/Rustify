fn special_filter(nums: &[i32]) -> i32 {
    let mut num = 0;
    for &value in nums.iter() {
        if value > 10 {
            let buffer = value.to_string();
            let len = buffer.len();
            if buffer.chars().next().unwrap().to_digit(10).unwrap() % 2 == 1
                && buffer.chars().last().unwrap().to_digit(10).unwrap() % 2 == 1
            {
                num += 1;
            }
        }
    }
    num
}

fn main() {
    let nums = vec![11, 22, 33, 44, 55];
    let result = special_filter(&nums);
    println!("Number of special filtered elements: {}", result);
}