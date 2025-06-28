fn even_odd_palindrome(n: i32) -> (i32, i32) {
    let mut even_count = 0;
    let mut odd_count = 0;

    for i in 1..=n {
        let w = i.to_string();
        let len = w.len();
        let is_palindrome = w.chars().zip(w.chars().rev()).take(len / 2).all(|(a, b)| a == b);

        if is_palindrome {
            if i % 2 == 0 {
                even_count += 1;
            } else {
                odd_count += 1;
            }
        }
    }

    (even_count, odd_count)
}

fn main() {
    let n = 1000;
    let (even_count, odd_count) = even_odd_palindrome(n);
    println!("Even palindrome count: {}", even_count);
    println!("Odd palindrome count: {}", odd_count);
}