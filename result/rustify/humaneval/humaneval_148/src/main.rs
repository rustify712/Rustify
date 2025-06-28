/// 返回两个行星之间的所有行星名称
///
/// # 参数
/// - `planet1`: 第一个行星的名称
/// - `planet2`: 第二个行星的名称
///
/// # 返回值
/// 返回一个包含两个行星之间所有行星名称的 `Vec<String>`，如果任一行星不在列表中，则返回 `None`。
fn bf(planet1: &str, planet2: &str) -> Option<Vec<String>> {
    let planets = vec![
        "Mercury".to_string(),
        "Venus".to_string(),
        "Earth".to_string(),
        "Mars".to_string(),
        "Jupiter".to_string(),
        "Saturn".to_string(),
        "Uranus".to_string(),
        "Neptune".to_string(),
    ];

    let pos1 = planets.iter().position(|p| p == planet1)?;
    let pos2 = planets.iter().position(|p| p == planet2)?;

    let (start, end) = if pos1 < pos2 { (pos1, pos2) } else { (pos2, pos1) };

    Some(planets[start + 1..end].to_vec())
}