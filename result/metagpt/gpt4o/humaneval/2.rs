fn truncate_number(number: f32) -> f32 {
    number - number.floor()
}

fn main() {
    let number = 5.67;
    let truncated = truncate_number(number);
    println!("Truncated number: {}", truncated);
}