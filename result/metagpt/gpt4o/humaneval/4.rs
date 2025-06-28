fn mean_absolute_deviation(numbers: &[f32]) -> f32 {
    let sum: f32 = numbers.iter().sum();
    let avg = sum / numbers.len() as f32;

    let msum: f32 = numbers.iter().map(|&x| (x - avg).abs()).sum();
    msum / numbers.len() as f32
}

fn main() {
    let numbers = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mad = mean_absolute_deviation(&numbers);
    println!("Mean Absolute Deviation: {}", mad);
}