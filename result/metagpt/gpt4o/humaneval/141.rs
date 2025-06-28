fn file_name_check(file_name: &str) -> &'static str {
    let length = file_name.len();
    if length < 5 {
        return "No";
    }

    let first_char = file_name.chars().next().unwrap();
    if !first_char.is_alphabetic() {
        return "No";
    }

    let last_four = &file_name[length - 4..];
    if last_four != ".txt" && last_four != ".exe" && last_four != ".dll" {
        return "No";
    }

    let numdigit = file_name.chars().filter(|&ch| ch.is_digit(10)).count();
    let numdot = file_name.chars().filter(|&ch| ch == '.').count();

    if numdigit > 3 || numdot != 1 {
        return "No";
    }

    "Yes"
}

fn main() {
    let file_name = "example.txt";
    let result = file_name_check(file_name);
    println!("File name check result: {}", result);
}