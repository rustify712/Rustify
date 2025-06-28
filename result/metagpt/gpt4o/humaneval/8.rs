fn sum_product(numbers: &[i32]) -> (i32, i32) {
    let mut sum = 0;
    let mut product = 1;

    for &number in numbers.iter() {
        sum += number;
        product *= number;
    }

    (sum, product)
}

fn main() {
    let numbers = vec![1, 2, 3, 4, 5];
    let (sum, product) = sum_product(&numbers);
    println!("Sum: {}, Product: {}", sum, product);
}