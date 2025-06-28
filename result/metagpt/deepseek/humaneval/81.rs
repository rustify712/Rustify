pub fn numerical_letter_grade(grades: &[f32]) -> Vec<String> {
    grades.iter().map(|&grade| {
        if grade >= 3.9999 {
            "A+".to_string()
        } else if grade > 3.7001 && grade < 3.9999 {
            "A".to_string()
        } else if grade > 3.3001 && grade <= 3.7001 {
            "A-".to_string()
        } else if grade > 3.0001 && grade <= 3.3001 {
            "B+".to_string()
        } else if grade > 2.7001 && grade <= 3.0001 {
            "B".to_string()
        } else if grade > 2.3001 && grade <= 2.7001 {
            "B-".to_string()
        } else if grade > 2.0001 && grade <= 2.3001 {
            "C+".to_string()
        } else if grade > 1.7001 && grade <= 2.0001 {
            "C".to_string()
        } else if grade > 1.3001 && grade <= 1.7001 {
            "C-".to_string()
        } else if grade > 1.0001 && grade <= 1.3001 {
            "D+".to_string()
        } else if grade > 0.7001 && grade <= 1.0001 {
            "D".to_string()
        } else if grade > 0.0001 && grade <= 0.7001 {
            "D-".to_string()
        } else {
            "E".to_string()
        }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_numerical_letter_grade() {
        assert_eq!(numerical_letter_grade(&[4.0]), vec!["A+"]);
        assert_eq!(numerical_letter_grade(&[3.8]), vec!["A"]);
        assert_eq!(numerical_letter_grade(&[3.5]), vec!["A-"]);
        assert_eq!(numerical_letter_grade(&[0.0]), vec!["E"]);
    }
}