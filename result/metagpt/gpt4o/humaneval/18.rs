fn how_many_times(s: &str, substring: &str) -> usize {
    let mut count = 0;
    let str_len = s.len();
    let sub_len = substring.len();

    if str_len == 0 || sub_len == 0 {
        return 0;
    }

    for i in 0..=str_len - sub_len {
        if &s[i..i + sub_len] == substring {
            count += 1;
        }
    }

    count
}

fn main() {
    let s = "hello hello world";
    let substring = "hello";
    let occurrences = how_many_times(s, substring);
    println!("Occurrences of '{}': {}", substring, occurrences);
}