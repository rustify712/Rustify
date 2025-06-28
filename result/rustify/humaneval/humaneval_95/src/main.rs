use std::collections::HashMap;

/// 检查字典中的所有键是否都是纯小写或纯大写字母组成，并且所有键的大小写形式必须一致。
/// 如果字典为空，返回 `false`；如果键中包含非字母字符，返回 `false`；如果键中同时包含大写和小写字母，返回 `false`；否则返回 `true`。
///
/// # 参数
/// - `dict`: 要检查的字典，键和值都是字符串。
///
/// # 返回值
/// - `bool`: 如果字典中的所有键都是纯小写或纯大写字母组成，并且所有键的大小写形式一致，返回 `true`；否则返回 `false`。
fn check_dict_case(dict: &HashMap<String, String>) -> bool {
    if dict.is_empty() {
        return false;
    }

    for (key, _) in dict.iter() {
        let mut is_lower = false;
        let mut is_upper = false;

        for c in key.chars() {
            if !c.is_ascii_alphabetic() {
                return false;
            }
            if c.is_ascii_lowercase() {
                is_lower = true;
            }
            if c.is_ascii_uppercase() {
                is_upper = true;
            }
            if is_lower && is_upper {
                return false;
            }
        }
    }

    true
}
