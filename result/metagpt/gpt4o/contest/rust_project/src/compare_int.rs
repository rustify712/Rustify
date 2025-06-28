// compare_int.rs

/// Comparison functions for integers in Rust.
///
/// These functions provide similar functionality to the C functions for
/// comparing integers.

/// Checks if two integers are equal.
///
/// # Arguments
/// * `int1` - A reference to the first integer.
/// * `int2` - A reference to the second integer.
///
/// # Returns
/// `true` if the integers are equal, `false` otherwise.
pub fn int_equal(int1: &i32, int2: &i32) -> bool {
    int1 == int2
}

/// Compares two integers.
///
/// # Arguments
/// * `int1` - A reference to the first integer.
/// * `int2` - A reference to the second integer.
///
/// # Returns
/// - `-1` if `int1` is less than `int2`
/// - `1` if `int1` is greater than `int2`
/// - `0` if both are equal
pub fn int_compare(int1: &i32, int2: &i32) -> i32 {
    if int1 < int2 {
        -1
    } else if int1 > int2 {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int_equal() {
        assert!(int_equal(&5, &5));
        assert!(!int_equal(&5, &10));
    }

    #[test]
    fn test_int_compare() {
        assert_eq!(int_compare(&5, &10), -1);
        assert_eq!(int_compare(&10, &5), 1);
        assert_eq!(int_compare(&5, &5), 0);
    }
}