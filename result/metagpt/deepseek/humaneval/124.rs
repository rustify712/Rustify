pub fn valid_date(date: &str) -> bool {
    if date.len() != 10 {
        return false;
    }
    
    let chars: Vec<char> = date.chars().collect();
    for i in 0..10 {
        match i {
            2 | 5 => {
                if chars[i] != '-' {
                    return false;
                }
            },
            _ => {
                if !chars[i].is_ascii_digit() {
                    return false;
                }
            }
        }
    }
    
    let mm = date[0..2].parse::<u32>().unwrap_or(0);
    let dd = date[3..5].parse::<u32>().unwrap_or(0);
    let _yy = date[6..10].parse::<u32>().unwrap_or(0);
    
    if mm < 1 || mm > 12 {
        return false;
    }
    if dd < 1 || dd > 31 {
        return false;
    }
    
    match mm {
        4 | 6 | 9 | 11 if dd > 30 => return false,
        2 if dd > 29 => return false,
        _ => ()
    }
    
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_date() {
        assert!(valid_date("03-11-2023"));
        assert!(!valid_date("02-30-2023"));
        assert!(!valid_date("13-01-2023"));
    }
}