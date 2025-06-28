/// Converts an integer `x` to a string representation in the given `base`.
///
/// # Arguments
/// * `x` - The integer to convert.
/// * `base` - The base to convert to (must be between 2 and 36).
///
/// # Returns
/// Returns a `Result` containing the string representation of `x` in the given `base`,
/// or an error if the base is invalid.
fn change_base(x: i32, base: i32) -> Result<String, &'static str> {
    if base < 2 || base > 36 {
        return Err("Base must be between 2 and 36");
    }

    let mut out = String::new();
    let mut x = x;
    while x > 0 {
        out = format!("{}{}", x % base, out);
        x /= base;
    }
    Ok(out)
}