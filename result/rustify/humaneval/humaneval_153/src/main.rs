/// 计算并返回强度最大的扩展名与类名的组合。
///
/// # 参数
/// - `class_name`: 类名，类型为 `&str`。
/// - `extensions`: 扩展名列表，类型为 `Vec<String>`。
///
/// # 返回值
/// 返回一个 `String`，表示类名与强度最大的扩展名的组合。
fn strongest_extension(class_name: &str, extensions: Vec<String>) -> String {
    let mut strongest = String::new();
    let mut max = -1000;

    for extension in extensions {
        let mut strength = 0;
        for chr in extension.chars() {
            if chr.is_ascii_uppercase() {
                strength += 1;
            } else if chr.is_ascii_lowercase() {
                strength -= 1;
            }
        }
        if strength > max {
            max = strength;
            strongest = extension;
        }
    }

    format!("{}.{}", class_name, strongest)
}