/// Rounds a string representation of a floating-point number to the nearest integer.
///
/// # Arguments
///
/// * `value` - A string slice that holds the floating-point number.
///
/// # Returns
///
/// Returns the nearest integer as a `Result<i32, std::num::ParseFloatError>`.
/// If the string cannot be parsed as a floating-point number, an error is returned.
fn closest_integer(value: &str) -> Result<i32, std::num::ParseFloatError> {
    let w: f64 = value.parse()?;
    Ok(w.round() as i32)
}