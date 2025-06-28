/// Converts a decimal integer to a binary string with 'db' prefix and suffix.
///
/// # Arguments
/// * `decimal` - The decimal integer to convert.
///
/// # Returns
/// A string representing the binary equivalent of the input decimal number,
/// prefixed and suffixed with 'db'.
fn decimal_to_binary(decimal: i32) -> String {
    if decimal == 0 {
        return "db0db".to_string();
    }

    let mut out = String::new();
    let mut num = decimal;

    while num > 0 {
        out = format!("{}{}", num % 2, out);
        num /= 2;
    }

    format!("db{}db", out)
}
