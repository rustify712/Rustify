use std::vec::Vec;

struct StringArray {
    data: Vec<String>,
}

impl StringArray {
    fn new(capacity: usize) -> Self {
        StringArray {
            data: Vec::with_capacity(capacity),
        }
    }

    fn push_back(&mut self, str: &str) {
        self.data.push(str.to_string());
    }
}

fn separate_paren_groups(paren_string: &str) -> StringArray {
    let mut all_parens = StringArray::new(10); // Initial capacity of 10
    let mut current_paren = String::with_capacity(100); // Assume each group is less than 100 characters
    let mut level = 0;

    for chr in paren_string.chars() {
        if chr == ' ' {
            continue; // Ignore spaces
        }
        if chr == '(' {
            level += 1;
            current_paren.push(chr);
        }
        if chr == ')' {
            level -= 1;
            current_paren.push(chr);
            if level == 0 {
                all_parens.push_back(&current_paren);
                current_paren.clear(); // Reset current group
            }
        }
    }

    all_parens
}

fn main() {
    let paren_string = "(()) (()) ((()))";
    let result = separate_paren_groups(paren_string);
    for group in result.data {
        println!("{}", group);
    }
}