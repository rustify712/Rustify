fn unique_digits(x: &[i32]) -> Vec<i32> {
    let mut out = Vec::new();

    for &num in x.iter() {
        let mut num_copy = num;
        let mut unique = true;

        if num_copy == 0 {
            unique = false;
        }

        while num_copy > 0 && unique {
            if num_copy % 2 == 0 {
                unique = false;
            }
            num_copy /= 10;
        }

        if unique {
            out.push(num);
        }
    }

    out.sort();
    out
}

fn main() {
    let numbers = vec![123, 456, 789, 135, 246];
    let result = unique_digits(&numbers);
    println!("Numbers with unique odd digits: {:?}", result);
}