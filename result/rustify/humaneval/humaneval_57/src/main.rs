/// Checks if a vector of floats is monotonic.
///
/// # Arguments
///
/// * `l` - A reference to a vector of floats.
///
/// # Returns
///
/// Returns `true` if the vector is monotonic (either increasing or decreasing),
/// otherwise returns `false`.
fn monotonic(l: &Vec<f32>) -> bool {
    let mut incr = false;
    let mut decr = false;

    for (prev, curr) in l.iter().zip(l.iter().skip(1)) {
        if curr > prev {
            incr = true;
        }
        if curr < prev {
            decr = true;
        }
    }

    !(incr && decr)
}