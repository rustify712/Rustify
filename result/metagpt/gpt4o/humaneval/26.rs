use std::collections::HashSet;

fn remove_duplicates(numbers: &[i32]) -> Vec<i32> {
    let mut has1 = HashSet::new();
    let mut has2 = HashSet::new();

    for &number in numbers {
        if has1.contains(&number) {
            has2.insert(number);
        } else {
            has1.insert(number);
        }
    }

    numbers.iter().filter(|&&number| !has2.contains(&number)).cloned().collect()
}

fn main() {
    let numbers = vec![1, 2, 3, 2, 4, 5, 3, 6];
    let result = remove_duplicates(&numbers);
    println!("Numbers after removing duplicates: {:?}", result);
}