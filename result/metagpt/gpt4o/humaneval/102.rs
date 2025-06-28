fn choose_num(x: i32, y: i32) -> i32 {
    if y < x {
        return -1;
    }
    if y == x && y % 2 == 1 {
        return -1;
    }
    if y % 2 == 1 {
        return y - 1;
    }
    y
}

fn main() {
    let x = 5;
    let y = 7;
    let result = choose_num(x, y);
    println!("Result: {}", result);
}