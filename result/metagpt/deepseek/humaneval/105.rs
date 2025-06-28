pub fn by_length(arr: &[i32]) -> Vec<String> {
    let mut filtered: Vec<i32> = arr.iter()
        .filter(|&&x| x >= 1 && x <= 9)
        .cloned()
        .collect();
    
    filtered.sort_unstable();
    filtered.reverse();
    
    filtered.iter().map(|&num| {
        match num {
            1 => "One".to_string(),
            2 => "Two".to_string(),
            3 => "Three".to_string(),
            4 => "Four".to_string(),
            5 => "Five".to_string(),
            6 => "Six".to_string(),
            7 => "Seven".to_string(),
            8 => "Eight".to_string(),
            9 => "Nine".to_string(),
            _ => "".to_string(),
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_by_length() {
        assert_eq!(by_length(&[1, 2, 3]), vec!["Three", "Two", "One"]);
        assert_eq!(by_length(&[9, 8, 7]), vec!["Nine", "Eight", "Seven"]);
        assert_eq!(by_length(&[0, 10]), Vec::<String>::new());
    }
}