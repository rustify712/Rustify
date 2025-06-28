pub fn max_element(l: &[f32]) -> f32 {
    l.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_element() {
        assert_eq!(max_element(&[1.0, 2.0, 3.0]), 3.0);
        assert_eq!(max_element(&[-1.0, -2.0, -3.0]), -1.0);
    }
}