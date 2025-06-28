// compare_pointer.rs

/// Comparison functions for generic pointers in Rust.
/// 
/// In Rust, we use raw pointers for similar functionality as C void pointers.
/// However, Rust's safety guarantees require us to handle these pointers carefully.

/// Checks if two raw pointers are equal.
///
/// # Safety
/// This function is unsafe because it dereferences raw pointers.
pub unsafe fn pointer_equal<T>(location1: *const T, location2: *const T) -> bool {
    location1 == location2
}

/// Compares two raw pointers.
///
/// # Safety
/// This function is unsafe because it dereferences raw pointers.
/// 
/// # Returns
/// - `-1` if `location1` is less than `location2`
/// - `1` if `location1` is greater than `location2`
/// - `0` if both are equal
pub unsafe fn pointer_compare<T>(location1: *const T, location2: *const T) -> i32 {
    if location1 < location2 {
        -1
    } else if location1 > location2 {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pointer_equal() {
        let a = 5;
        let b = 5;
        let ptr_a: *const i32 = &a;
        let ptr_b: *const i32 = &b;
        let ptr_a_clone: *const i32 = &a;

        unsafe {
            assert!(pointer_equal(ptr_a, ptr_a_clone));
            assert!(!pointer_equal(ptr_a, ptr_b));
        }
    }

    #[test]
    fn test_pointer_compare() {
        let a = 5;
        let b = 10;
        let ptr_a: *const i32 = &a;
        let ptr_b: *const i32 = &b;

        unsafe {
            assert_eq!(pointer_compare(ptr_a, ptr_a), 0);
            assert_eq!(pointer_compare(ptr_a, ptr_b), -1);
            assert_eq!(pointer_compare(ptr_b, ptr_a), 1);
        }
    }
}