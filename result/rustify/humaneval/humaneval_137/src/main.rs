use std::str::FromStr;

#[derive(Debug, PartialEq, PartialOrd)]
enum Value {
    Int(i32),
    Double(f64),
    String(String),
}

impl Value {
    fn to_double(&self) -> Result<f64, String> {
        match self {
            Value::Int(i) => Ok(*i as f64),
            Value::Double(d) => Ok(*d),
            Value::String(s) => {
                let s = s.replace(',', ".");
                f64::from_str(&s).map_err(|_| "Failed to parse string as double".to_string())
            }
        }
    }
}

fn compare_one(a: Value, b: Value) -> Result<Value, String> {
    let numa = a.to_double()?;
    let numb = b.to_double()?;

    if numa == numb {
        Ok(Value::String("None".to_string()))
    } else if numa < numb {
        Ok(b)
    } else {
        Ok(a)
    }
}
