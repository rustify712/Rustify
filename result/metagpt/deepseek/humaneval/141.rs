pub fn file_name_check(file_name: &str) -> &'static str {
    if file_name.len() < 5 {
        return "No";
    }
    
    let first_char = file_name.chars().next().unwrap();
    if !first_char.is_ascii_alphabetic() {
        return "No";
    }
    
    if !file_name.ends_with(".txt") && !file_name.ends_with(".exe") && !file_name.ends_with(".dll") {
        return "No";
    }
    
    let num_digit = file_name.chars().filter(|c| c.is_ascii_digit()).count();
    let num_dot = file_name.chars().filter(|&c| c == '.').count();
    
    if num_digit > 3 || num_dot != 1 {
        return "No";
    }
    
    "Yes"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_name_check() {
        assert_eq!(file_name_check("test.txt"), "Yes");
        assert_eq!(file_name_check("1test.exe"), "No");
        assert_eq!(file_name_check("test.dll.txt"), "No");
    }
}