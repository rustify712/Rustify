use std::collections::HashMap;

pub fn check_dict_case(dict: &HashMap<String, String>) -> bool {
    if dict.is_empty() {
        return false;
    }
    
    let mut has_upper = false;
    let mut has_lower = false;
    
    for key in dict.keys() {
        if !key.chars().all(|c| c.is_ascii_alphabetic()) {
            return false;
        }
        
        if key.chars().any(|c| c.is_ascii_uppercase()) {
            has_upper = true;
        }
        if key.chars().any(|c| c.is_ascii_lowercase()) {
            has_lower = true;
        }
        
        if has_upper && has_lower {
            return false;
        }
    }
    
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_check_dict_case() {
        let mut dict1 = HashMap::new();
        dict1.insert("KEY".to_string(), "VALUE".to_string());
        assert!(check_dict_case(&dict1));
        
        let mut dict2 = HashMap::new();
        dict2.insert("key".to_string(), "value".to_string());
        assert!(check_dict_case(&dict2));
        
        let mut dict3 = HashMap::new();
        dict3.insert("Key".to_string(), "Value".to_string());
        assert!(!check_dict_case(&dict3));
    }
}