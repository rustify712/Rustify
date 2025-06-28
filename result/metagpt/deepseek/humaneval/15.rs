pub fn string_sequence(n: i32) -> String {
    let mut result = String::from("0");
    for i in 1..=n {
        result.push_str(&format!(" {}", i));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_string_sequence() {
        assert_eq!(string_sequence(3), "0 1 2 3");
        assert_eq!(string_sequence(0), "0");
    }
}