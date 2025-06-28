fn valid_date(date: &str) -> bool {
    if date.len() != 10 {
        return false;
    }

    for (i, ch) in date.chars().enumerate() {
        if (i == 2 || i == 5) && ch != '-' {
            return false;
        } else if (i != 2 && i != 5) && !ch.is_digit(10) {
            return false;
        }
    }

    let mm: i32 = date[0..2].parse().unwrap_or(0);
    let dd: i32 = date[3..5].parse().unwrap_or(0);
    let yy: i32 = date[6..10].parse().unwrap_or(0);

    if mm < 1 || mm > 12 {
        return false;
    }
    if dd < 1 || dd > 31 {
        return false;
    }
    if dd == 31 && (mm == 4 || mm == 6 || mm == 9 || mm == 11 || mm == 2) {
        return false;
    }
    if dd == 30 && mm == 2 {
        return false;
    }

    true
}

fn main() {
    let date = "12-31-2023";
    let is_valid = valid_date(date);
    println!("Is the date valid? {}", is_valid);
}