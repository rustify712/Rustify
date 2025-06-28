use std::str::FromStr;

#[derive(Debug, Clone)]
pub enum AnyValue {
    Int(i32),
    Double(f64),
    String(String),
}

impl AnyValue {
    pub fn to_f64(&self) -> f64 {
        match self {
            AnyValue::Int(i) => *i as f64,
            AnyValue::Double(d) => *d,
            AnyValue::String(s) => {
                let s = s.replace(',', ".");
                f64::from_str(&s).unwrap_or(0.0)
            }
        }
    }
}

pub fn compare_one(a: AnyValue, b: AnyValue) -> Option<AnyValue> {
    let a_val = a.to_f64();
    let b_val = b.to_f64();
    
    if (a_val - b_val).abs() < f64::EPSILON {
        None
    } else if a_val > b_val {
        Some(a)
    } else {
        Some(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compare_one() {
        assert_eq!(
            compare_one(AnyValue::Int(1), AnyValue::Int(2)),
            Some(AnyValue::Int(2))
        );
        assert_eq!(
            compare_one(AnyValue::String("1,2".to_string()), AnyValue::Double(1.2)),
            None
        );
    }
}