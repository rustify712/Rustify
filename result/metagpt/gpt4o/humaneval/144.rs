fn simplify(x: &str, n: &str) -> bool {
    let x_parts: Vec<&str> = x.split('/').collect();
    let n_parts: Vec<&str> = n.split('/').collect();

    if x_parts.len() != 2 || n_parts.len() != 2 {
        return false; // Invalid input format
    }

    let a = x_parts[0].parse::<i32>().unwrap_or(0);
    let b = x_parts[1].parse::<i32>().unwrap_or(0);
    let c = n_parts[0].parse::<i32>().unwrap_or(0);
    let d = n_parts[1].parse::<i32>().unwrap_or(0);

    if b == 0 || d == 0 {
        return false; // Avoid division by zero
    }

    (a * c) % (b * d) == 0
}

fn main() {
    let x = "4/2";
    let n = "2/1";
    let result = simplify(x, n);
    println!("Can simplify: {}", result);
}