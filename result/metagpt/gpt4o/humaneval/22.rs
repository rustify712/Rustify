enum Any {
    Int(i32),
    Double(f64),
    String(String),
    Other(Box<dyn std::any::Any>),
}

struct ListAny {
    items: Vec<Any>,
}

fn filter_integers(values: &ListAny) -> Vec<i32> {
    let mut out = Vec::new();

    for item in &values.items {
        if let Any::Int(value) = item {
            out.push(*value);
        }
    }

    out
}

fn main() {
    let values = ListAny {
        items: vec![
            Any::Int(42),
            Any::Double(3.14),
            Any::String(String::from("hello")),
            Any::Int(7),
        ],
    };

    let integers = filter_integers(&values);
    println!("Filtered integers: {:?}", integers);
}