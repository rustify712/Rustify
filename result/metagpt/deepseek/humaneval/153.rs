pub fn strongest_extension(class_name: &str, extensions: &[&str]) -> String {
    let (strongest, _) = extensions.iter()
        .map(|ext| {
            let strength = ext.chars().fold(0, |acc, c| {
                if c.is_ascii_uppercase() { acc + 1 }
                else if c.is_ascii_lowercase() { acc - 1 }
                else { acc }
            });
            (ext, strength)
        })
        .max_by_key(|&(_, strength)| strength)
        .unwrap();
    
    format!("{}.{}", class_name, strongest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strongest_extension() {
        assert_eq!(
            strongest_extension("Test", &["A", "b", "Cc"]),
            "Test.Cc"
        );
        assert_eq!(
            strongest_extension("MyClass", &["AA", "Be", "CC"]),
            "MyClass.AA"
        );
    }
}