/// Encrypts a string by shifting each character by 4 positions in the alphabet.
///
/// # Arguments
///
/// * `s` - The input string to be encrypted.
///
/// # Returns
///
/// Returns a new `String` containing the encrypted result.
fn encrypt(s: &str) -> String {
    s.chars()
        .map(|c| {
            let w = ((c as u8 + 4 - b'a') % 26 + b'a') as char;
            w
        })
        .collect()
}