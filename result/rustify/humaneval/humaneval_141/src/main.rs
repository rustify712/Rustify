/// Checks if the given file name meets the specified criteria.
///
/// # Arguments
/// * `file_name` - A string slice that holds the file name to be checked.
///
/// # Returns
/// Returns `"Yes"` if the file name meets all the criteria, otherwise returns `"No"`.
fn file_name_check(file_name: &str) -> &'static str {
    if file_name.len() < 5 {
        return "No";
    }

    let first_char = file_name.chars().next().unwrap();
    if !first_char.is_ascii_alphabetic() {
        return "No";
    }

    if !(file_name.ends_with(".txt") || file_name.ends_with(".exe") || file_name.ends_with(".dll")) {
        return "No";
    }

    let num_digits = file_name.chars().filter(|c| c.is_ascii_digit()).count();
    let num_dots = file_name.chars().filter(|c| *c == '.').count();

    if num_digits > 3 || num_dots != 1 {
        return "No";
    }

    "Yes"
}