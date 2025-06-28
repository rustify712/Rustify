#[derive(Debug, Clone)]
enum ValueType {
    Int,
    Double,
    String,
}

#[derive(Debug, Clone)]
struct AnyValue {
    value_type: ValueType,
    int_value: Option<i32>,
    double_value: Option<f64>,
    string_value: Option<String>,
}

fn compare_one(a: AnyValue, b: AnyValue) -> AnyValue {
    let numa = match a.value_type {
        ValueType::String => {
            let mut s = a.string_value.unwrap();
            if let Some(comma_pos) = s.find(',') {
                s.replace_range(comma_pos..=comma_pos, ".");
            }
            s.parse::<f64>().unwrap_or(0.0)
        }
        ValueType::Int => a.int_value.unwrap() as f64,
        ValueType::Double => a.double_value.unwrap(),
    };

    let numb = match b.value_type {
        ValueType::String => {
            let mut s = b.string_value.unwrap();
            if let Some(comma_pos) = s.find(',') {
                s.replace_range(comma_pos..=comma_pos, ".");
            }
            s.parse::<f64>().unwrap_or(0.0)
        }
        ValueType::Int => b.int_value.unwrap() as f64,
        ValueType::Double => b.double_value.unwrap(),
    };

    if numa == numb {
        AnyValue {
            value_type: ValueType::String,
            int_value: None,
            double_value: None,
            string_value: Some("None".to_string()),
        }
    } else if numa < numb {
        b
    } else {
        a
    }
}

fn main() {
    let a = AnyValue {
        value_type: ValueType::String,
        int_value: None,
        double_value: None,
        string_value: Some("123,45".to_string()),
    };

    let b = AnyValue {
        value_type: ValueType::Double,
        int_value: None,
        double_value: Some(123.46),
        string_value: None,
    };

    let result = compare_one(a.clone(), b.clone());
    println!("Larger value: {:?}", result);
}