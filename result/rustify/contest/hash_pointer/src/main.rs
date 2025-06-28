/// Generate a hash key for a pointer. The value pointed at by the pointer
/// is not used, only the pointer itself.
///
/// # Arguments
///
/// * `location` - The pointer
///
/// # Returns
///
/// A hash key for the pointer.
pub fn pointer_hash(location: *const ()) -> u32 {
    (location as usize) as u32
}