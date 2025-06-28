/// Validates a date string in the format `MM-DD-YYYY`.
///
/// # Arguments
/// * `date` - A string slice that holds the date in `MM-DD-YYYY` format.
///
/// # Returns
/// Returns `true` if the date is valid, otherwise `false`.
fn valid_date(date: &str) -> bool {
    if date.len() != 10 {
        return false;
    }

    for (i, c) in date.chars().enumerate() {
        match i {
            2 | 5 => {
                if c != '-' {
                    return false;
                }
            }
            _ => {
                if !c.is_ascii_digit() {
                    return false;
                }
            }
        }
    }

    let mm = date[0..2].parse::<u32>().unwrap_or(0);
    let dd = date[3..5].parse::<u32>().unwrap_or(0);
    let yy = date[6..10].parse::<u32>().unwrap_or(0);

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