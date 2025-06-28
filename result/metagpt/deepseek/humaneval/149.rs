pub fn sorted_list_sum(lst: Vec<&str>) -> Vec<String> {
    let mut result: Vec<String> = lst.into_iter()
        .filter(|s| s.len() % 2 == 0)
        .map(|s| s.to_string())
        .collect();
    
    result.sort_by(|a, b| {
        if a.len() != b.len() {
            a.len().cmp(&b.len())
        } else {
            a.cmp(b)
        }
    });
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sorted_list_sum() {
        assert_eq!(
            sorted_list_sum(vec!["aa", "a", "aaa"]), 
            vec!["aa"]
        );
        assert_eq!(
            sorted_list_sum(vec!["ab", "cd", "ef"]), 
            vec!["ab", "cd", "ef"]
        );
    }
}