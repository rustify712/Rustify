pub enum Any {
    Int(i32),
    Double(f64),
    String(String),
    Other(Box<dyn std::any::Any>),
}

pub fn filter_integers(values: &[Any]) -> Vec<i32> {
    values
        .iter()
        .filter_map(|item| {
            if let Any::Int(value) = item {
                Some(*value)
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_integers() {
        let values = vec![Any::Int(1), Any::Double(2.0), Any::Int(3), Any::String("4".to_string())];
        assert_eq!(filter_integers(&values), vec![1, 3]);
    }
}