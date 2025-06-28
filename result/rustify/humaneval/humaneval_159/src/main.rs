/// Consumes `number`, `need`, and `remaining` to calculate the result.
///
/// # Arguments
///
/// * `number` - The initial number.
/// * `need` - The amount needed.
/// * `remaining` - The remaining amount.
///
/// # Returns
///
/// A tuple containing the sum of `number` and the consumed amount, and the remaining amount after consumption.
fn eat(number: i32, need: i32, remaining: i32) -> (i32, i32) {
    if need > remaining {
        (number + remaining, 0)
    } else {
        (number + need, remaining - need)
    }
}