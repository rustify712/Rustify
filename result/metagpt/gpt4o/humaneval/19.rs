fn sort_numbers(numbers: &str) -> String {
    // Define the mapping from string to number
    let tonum_keys = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
    let mut count = [0; 10];

    // Split the input string and count occurrences of each number
    for token in numbers.split_whitespace() {
        if let Some(pos) = tonum_keys.iter().position(|&key| key == token) {
            count[pos] += 1;
        }
    }

    // Build the output string
    let mut out = String::new();
    for (i, &num) in count.iter().enumerate() {
        for _ in 0..num {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(tonum_keys[i]);
        }
    }

    out
}

fn main() {
    let numbers = "three one four one five nine";
    let sorted_numbers = sort_numbers(numbers);
    println!("Sorted numbers: {}", sorted_numbers);
}