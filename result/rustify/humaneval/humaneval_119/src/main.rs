/// Checks if the concatenation of two strings forms a valid parentheses match.
///
/// # Arguments
/// * `lst` - A vector containing two strings to be concatenated and checked.
///
/// # Returns
/// Returns "Yes" if the concatenation forms a valid parentheses match, otherwise returns "No".
fn match_parens(lst: Vec<String>) -> &'static str {
    let check_balance = |s: &String| -> bool {
        let mut count = 0;
        for c in s.chars() {
            match c {
                '(' => count += 1,
                ')' => count -= 1,
                _ => (),
            }
            if count < 0 {
                return false;
            }
        }
        count == 0
    };

    let l1 = lst[0].clone() + &lst[1];
    if check_balance(&l1) {
        return "Yes";
    }

    let l2 = lst[1].clone() + &lst[0];
    if check_balance(&l2) {
        return "Yes";
    }

    "No"
}