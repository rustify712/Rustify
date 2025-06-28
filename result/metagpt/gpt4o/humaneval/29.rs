struct StringArray {
    data: Vec<String>,
}

impl StringArray {
    fn new() -> Self {
        StringArray { data: Vec::new() }
    }

    fn filter_by_prefix(&self, prefix: &str) -> StringArray {
        let mut out = StringArray::new();
        for s in &self.data {
            if s.starts_with(prefix) {
                out.data.push(s.clone());
            }
        }
        out
    }
}

fn main() {
    let strings = StringArray {
        data: vec![
            "apple".to_string(),
            "apricot".to_string(),
            "banana".to_string(),
            "avocado".to_string(),
        ],
    };

    let prefix = "ap";
    let filtered_strings = strings.filter_by_prefix(prefix);
    println!("Filtered strings: {:?}", filtered_strings.data);
}