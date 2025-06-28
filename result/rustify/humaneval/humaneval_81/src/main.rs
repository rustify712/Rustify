/// Converts a vector of numerical grades to a vector of letter grades.
///
/// # Arguments
///
/// * `grades` - A slice of f32 values representing numerical grades.
///
/// # Returns
///
/// A vector of strings representing the corresponding letter grades.
fn numerical_letter_grade(grades: &[f32]) -> Vec<String> {
    let mut out = Vec::new();
    for &grade in grades {
        if grade >= 3.9999 {
            out.push("A+".to_string());
        } else if grade > 3.7001 && grade < 3.9999 {
            out.push("A".to_string());
        } else if grade > 3.3001 && grade <= 3.7001 {
            out.push("A-".to_string());
        } else if grade > 3.0001 && grade <= 3.3001 {
            out.push("B+".to_string());
        } else if grade > 2.7001 && grade <= 3.0001 {
            out.push("B".to_string());
        } else if grade > 2.3001 && grade <= 2.7001 {
            out.push("B-".to_string());
        } else if grade > 2.0001 && grade <= 2.3001 {
            out.push("C+".to_string());
        } else if grade > 1.7001 && grade <= 2.0001 {
            out.push("C".to_string());
        } else if grade > 1.3001 && grade <= 1.7001 {
            out.push("C-".to_string());
        } else if grade > 1.0001 && grade <= 1.3001 {
            out.push("D+".to_string());
        } else if grade > 0.7001 && grade <= 1.0001 {
            out.push("D".to_string());
        } else if grade > 0.0001 && grade <= 0.7001 {
            out.push("D-".to_string());
        } else if grade <= 0.0001 {
            out.push("E".to_string());
        }
    }
    out
}