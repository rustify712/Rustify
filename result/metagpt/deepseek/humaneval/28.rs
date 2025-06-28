pub fn concatenate(strings: &[&str]) -> String {
    strings.concat()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concatenate() {
        assert_eq!(concatenate(&["a", "b", "c"]), "abc");
        assert_eq!(concatenate(&[]), "");
    }
}