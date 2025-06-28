use regex::Regex;

pub fn fruit_distribution(s: &str, n: i32) -> i32 {
    let re = Regex::new(r"\d+").unwrap();
    let numbers: Vec<i32> = re.find_iter(s)
        .filter_map(|m| m.as_str().parse().ok())
        .collect();
    
    if numbers.len() >= 2 {
        n - numbers[0] - numbers[1]
    } else {
        n
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fruit_distribution() {
        assert_eq!(fruit_distribution("5 apples and 3 oranges", 10), 2);
        assert_eq!(fruit_distribution("10 apples", 20), 10);
        assert_eq!(fruit_distribution("no fruits", 5), 5);
    }
}