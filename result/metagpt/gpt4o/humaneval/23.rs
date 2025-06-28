fn my_strlen(s: &str) -> usize {
    s.chars().count()
}

fn main() {
    let test_str = "Hello, world!";
    let length = my_strlen(test_str);
    println!("Length of '{}': {}", test_str, length);
}