/// Checks if the product of two fractions is an integer.
///
/// # Arguments
///
/// * `x` - A string slice representing the first fraction (e.g., "1/2").
/// * `n` - A string slice representing the second fraction (e.g., "3/4").
///
/// # Returns
///
/// Returns `Ok(true)` if the product is an integer, `Ok(false)` otherwise.
/// Returns `Err` if the input strings are not in the correct format.
fn simplify(x: &str, n: &str) -> Result<bool, &'static str> {
    let parse_fraction = |s: &str| -> Result<(i32, i32), &'static str> {
        let parts: Vec<&str> = s.split('/').collect();
        if parts.len() != 2 {
            return Err("Invalid fraction format");
        }
        let numerator = parts[0].parse::<i32>().map_err(|_| "Invalid numerator")?;
        let denominator = parts[1].parse::<i32>().map_err(|_| "Invalid denominator")?;
        Ok((numerator, denominator))
    };

    let (a, b) = parse_fraction(x)?;
    let (c, d) = parse_fraction(n)?;

    Ok((a * c) % (b * d) == 0)
}