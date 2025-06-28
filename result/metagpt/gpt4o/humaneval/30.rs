struct FloatArray {
    data: Vec<f32>,
}

impl FloatArray {
    fn new(size: usize) -> Self {
        FloatArray {
            data: Vec::with_capacity(size),
        }
    }

    fn get_positive(&self) -> FloatArray {
        let mut out = FloatArray::new(0);
        for &value in &self.data {
            if value > 0.0 {
                out.data.push(value);
            }
        }
        out
    }
}

fn main() {
    let numbers = FloatArray {
        data: vec![1.0, -2.0, 3.5, 0.0, 4.0, -1.5],
    };

    let positive_numbers = numbers.get_positive();
    println!("Positive numbers: {:?}", positive_numbers.data);
}