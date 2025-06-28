/// Generate a hash key for a reference to an integer.
/// The value referenced is used to generate the key.
///
/// # Arguments
///
/// * `location` - A reference to an integer.
///
/// # Returns
///
/// A hash key for the value at the location.
pub fn int_hash(location: &i32) -> u32 {
    *location as u32
}