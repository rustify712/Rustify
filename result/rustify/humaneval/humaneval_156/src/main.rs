/// Converts an integer to a simplified Roman numeral representation.
///
/// # Arguments
/// * `number` - The integer to convert.
///
/// # Returns
/// A `String` representing the simplified Roman numeral.
fn int_to_mini_romank(number: i32) -> String {
    let mut current = String::new();
    let rep = vec!["m", "cm", "d", "cd", "c", "xc", "l", "xl", "x", "ix", "v", "iv", "i"];
    let num = vec![1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1];
    let mut pos = 0;
    let mut number = number;
    while number > 0 {
        while number >= num[pos] {
            current.push_str(rep[pos]);
            number -= num[pos];
        }
        if number > 0 {
            pos += 1;
        }
    }
    current
}