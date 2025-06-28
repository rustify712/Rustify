/// Compare two strings.
///
/// # Arguments
///
/// * `string1` - The first string.
/// * `string2` - The second string.
///
/// # Returns
///
/// A negative value if the first string should be sorted before the second,
/// a positive value if the first string should be sorted after the second,
/// zero if the two strings are equal.
pub fn string_compare(string1: &str, string2: &str) -> i32 {
    let result = string1.cmp(string2);
    match result {
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Equal => 0,
    }
}

/// Compare two strings, ignoring the case of letters.
///
/// # Arguments
///
/// * `string1` - The first string.
/// * `string2` - The second string.
///
/// # Returns
///
/// An `Ordering` value indicating the comparison result.
pub fn string_nocase_compare(string1: &str, string2: &str) -> std::cmp::Ordering {
    for (c1, c2) in string1.chars().zip(string2.chars()) {
        let cmp = c1.to_ascii_lowercase().cmp(&c2.to_ascii_lowercase());
        if cmp != std::cmp::Ordering::Equal {
            return cmp;
        }
    }
    string1.len().cmp(&string2.len())
}

/// Compare two strings to determine if they are equal, ignoring the case of letters.
///
/// # Arguments
///
/// * `string1` - The first string.
/// * `string2` - The second string.
///
/// # Returns
///
/// `true` if the strings are equal, `false` otherwise.
pub fn string_nocase_equal(string1: &str, string2: &str) -> bool {
    string1.eq_ignore_ascii_case(string2)
}