fn tri(n: usize) -> Vec<i32> {
    let mut out = vec![0; n + 1];
    out[0] = 1;
    if n == 0 {
        return out;
    }

    out[1] = 3;
    for i in 2..=n {
        if i % 2 == 0 {
            out[i] = 1 + (i / 2) as i32;
        } else {
            out[i] = out[i - 1] + out[i - 2] + 1 + ((i + 1) / 2) as i32;
        }
    }

    out
}

fn main() {
    let n = 5;
    let result = tri(n);
    println!("Tri sequence: {:?}", result);
}