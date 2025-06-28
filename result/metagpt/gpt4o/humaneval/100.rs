fn make_a_pile(n: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(n);
    out.push(n as i32);
    for i in 1..n {
        out.push(out[i - 1] + 2);
    }
    out
}

fn main() {
    let n = 5;
    let pile = make_a_pile(n);
    println!("{:?}", pile);
}