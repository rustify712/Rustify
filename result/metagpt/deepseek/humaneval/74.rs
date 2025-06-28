pub fn total_match(lst1: &[String], lst2: &[String]) -> Vec<String> {
    let total1: usize = lst1.iter().map(|s| s.len()).sum();
    let total2: usize = lst2.iter().map(|s| s.len()).sum();
    
    if total1 > total2 {
        lst2.to_vec()
    } else {
        lst1.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_total_match() {
        let lst1 = vec!["a".to_string(), "bb".to_string()];
        let lst2 = vec!["ccc".to_string()];
        assert_eq!(total_match(&lst1, &lst2), lst1);
        
        let lst3 = vec!["dddd".to_string()];
        assert_eq!(total_match(&lst1, &lst3), lst1);
    }
}