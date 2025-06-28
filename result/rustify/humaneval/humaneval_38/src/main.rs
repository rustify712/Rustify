/// Encodes a string by cycling groups of three characters.
///
/// # Arguments
///
/// * `s` - The input string to be encoded.
///
/// # Returns
///
/// Returns the encoded string.
fn encode_cyclic(s: &str) -> String {
    let mut output = String::new();
    let mut chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    for i in (0..len).step_by(3) {
        let end = std::cmp::min(i + 3, len);
        let mut group: Vec<char> = chars[i..end].to_vec();
        if group.len() == 3 {
            group.rotate_left(1);
        }
        output.extend(group);
    }
    output
}

/// Decodes a string that was encoded using the `encode_cyclic` function.
/// The input string is divided into groups of 3 characters, and each group is cyclically shifted right by one character.
///
/// # Arguments
/// * `s` - The encoded string to be decoded.
///
/// # Returns
/// The decoded string.
fn decode_cyclic(s: &str) -> String {
    let mut output = String::new();
    let mut chars = s.chars().collect::<Vec<char>>();
    let len = chars.len();

    for i in (0..len).step_by(3) {
        if i + 3 <= len {
            // Cyclically shift the group right by one character
            let temp = chars[i + 2];
            chars[i + 2] = chars[i + 1];
            chars[i + 1] = chars[i];
            chars[i] = temp;
        }
        output.extend(&chars[i..std::cmp::min(i + 3, len)]);
    }

    output
}