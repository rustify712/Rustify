fn intersection(interval1: [i32; 2], interval2: [i32; 2]) -> &'static str {
    let inter1 = interval1[0].max(interval2[0]);
    let inter2 = interval1[1].min(interval2[1]);
    let l = inter2 - inter1;

    if l < 2 {
        return "NO";
    }

    for i in 2..=((l as f64).sqrt() as i32) {
        if l % i == 0 {
            return "NO";
        }
    }

    "YES"
}

fn main() {
    let interval1 = [3, 10];
    let interval2 = [5, 15];
    let result = intersection(interval1, interval2);
    println!("Intersection result: {}", result);
}