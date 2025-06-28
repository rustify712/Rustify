pub fn exchange(lst1: &[i32], lst2: &[i32]) -> &'static str {
    let even_count = lst1.iter().filter(|&&x| x % 2 == 0).count() + 
                    lst2.iter().filter(|&&x| x % 2 == 0).count();
    
    if even_count >= lst1.len() {
        "YES"
    } else {
        "NO"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exchange() {
        assert_eq!(exchange(&[1, 2, 3], &[4, 5, 6]), "YES");
        assert_eq!(exchange(&[1, 3, 5], &[2, 4, 6]), "NO");
    }
}