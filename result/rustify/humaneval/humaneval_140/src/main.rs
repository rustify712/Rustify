/// Replaces spaces in the input string with specific characters.
///
/// # Arguments
///
/// * `text` - The input string containing spaces to be replaced.
///
/// # Returns
///
/// A new string with spaces replaced according to the rules:
/// - Single space: '_'
/// - Two consecutive spaces: '__'
/// - Three or more consecutive spaces: '-'
fn fix_spaces(text: &str) -> String {
    let mut out = String::new();
    let mut spacelen = 0;

    for c in text.chars() {
        if c == ' ' {
            spacelen += 1;
        } else {
            match spacelen {
                1 => out.push('_'),
                2 => out.push_str("__"),
                _ if spacelen > 2 => out.push('-'),
                _ => {}
            }
            spacelen = 0;
            out.push(c);
        }
    }

    match spacelen {
        1 => out.push('_'),
        2 => out.push_str("__"),
        _ if spacelen > 2 => out.push('-'),
        _ => {}
    }

    out
}