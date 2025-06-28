fn rescale_to_unit(numbers: &mut [f32]) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;

    // Find the minimum and maximum values in the array
    for &number in numbers.iter() {
        if number < min {
            min = number;
        }
        if number > max {
            max = number;
        }
    }

    // Rescale the numbers to the range [0, 1]
    for number in numbers.iter_mut() {
        *number = (*number - min) / (max - min);
    }
}

fn main() {
    let mut numbers = vec![3.0, 6.0, 9.0, 12.0];
    rescale_to_unit(&mut numbers);
    println!("Rescaled numbers: {:?}", numbers);
}