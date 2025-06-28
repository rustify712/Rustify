pub fn even_odd_palindrome(n: i32) -> (i32, i32) {
    let mut even_count = 0;
    let mut odd_count = 0;
    
    for i in 1..=n {
        let s = i.to_string();
        if s.chars().eq(s.chars().rev()) {
            if i % 2 == 0 {
                even_count += 1;
            } else {
                odd_count += 1;
            }
        }
    }
    
    (even_count, odd_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_even_odd_palindrome() {
        assert_eq!(even_odd_palindrome(10), (1, 5)); // 1,3,5,7,9 are odd palindromes, 2,4,6,8,10 - only 2 is even palindrome
        assert_eq!(even_odd_palindrome(5), (1, 3)); // 1,3,5 are odd palindromes, 2 is even palindrome
    }
}