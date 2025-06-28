pub fn is_cube(a: i32) -> bool {
    let abs_a = a.abs();
    for i in 0..=abs_a {
        let cube = i * i * i;
        if cube == abs_a {
            return true;
        }
        if cube > abs_a {
            break;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_cube() {
        assert!(is_cube(8));
        assert!(is_cube(27));
        assert!(is_cube(-27));
        assert!(!is_cube(10));
    }
}