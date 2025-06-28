// hash_pointer.rs

/// Hash function for a generic pointer in Rust.
///
/// In Rust, we use raw pointers for similar functionality as C void pointers.
/// This function takes a raw pointer and returns its hash value.
///
/// # Safety
/// This function is unsafe because it deals with raw pointers.
///
/// # Arguments
/// * `location` - A raw pointer to the location.
///
/// # Returns
/// The hash value of the pointer.
pub unsafe fn pointer_hash<T>(location: *const T) -> u32 {
    location as usize as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pointer_hash() {
        let value = 42;
        let ptr: *const i32 = &value;

        unsafe {
            let hash_value = pointer_hash(ptr);
            assert_eq!(hash_value, ptr as usize as u32);
        }
    }
}