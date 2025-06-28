/// Encodes the given message by toggling the case of each character and shifting vowels by 2 in the ASCII table.
///
/// # Arguments
/// * `message` - A string slice that holds the message to be encoded.
///
/// # Returns
/// A `String` containing the encoded message.
fn encode(message: &str) -> String {
    let vowels = "aeiouAEIOU";
    let mut out = String::new();

    for c in message.chars() {
        let mut w = c;
        if w.is_ascii_lowercase() {
            w = w.to_ascii_uppercase();
        } else if w.is_ascii_uppercase() {
            w = w.to_ascii_lowercase();
        }
        if vowels.contains(w) {
            w = (w as u8 + 2) as char;
        }
        out.push(w);
    }

    out
}