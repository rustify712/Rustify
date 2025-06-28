/// Decodes a string that was encoded using the `encode_shift` function.
///
/// # Arguments
///
/// * `s` - A string slice that holds the encoded string.
///
/// # Returns
///
/// A `String` containing the decoded string.
fn decode_shift(s: &str) -> String {
    s.chars()
        .map(|c| {
            let w = ((c as u8) + 21 - b'a') % 26 + b'a';
            w as char
        })
        .collect()
}

/// Encodes a string by shifting every character by 5 positions in the alphabet.
///
/// # Arguments
/// * `s` - The input string to be encoded.
///
/// # Returns
/// The encoded string.
fn encode_shift(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_lowercase() {
                let w = ((c as u8 - b'a' + 5) % 26) + b'a';
                w as char
            } else {
                c
            }
        })
        .collect()
}