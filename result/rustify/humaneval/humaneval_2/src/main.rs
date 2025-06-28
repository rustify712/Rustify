/// Returns the fractional part of a floating-point number.
///
/// # Arguments
/// * `number` - A floating-point number.
///
/// # Returns
/// The fractional part of the number.
fn truncate_number(number: f64) -> f64 {
    number.fract()
}