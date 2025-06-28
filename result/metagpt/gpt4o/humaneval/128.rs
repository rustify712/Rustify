fn prod_signs(arr: &[i32]) -> i32 {
    if arr.is_empty() {
        return -32768;
    }
    let mut sum = 0;
    let mut prods = 1;
    for &value in arr.iter() {
        sum += value.abs();
        if value == 0 {
            prods = 0;
        }
        if value < 0 {
            prods = -prods;
        }
    }
    sum * prods
}

fn main() {
    let arr = vec![1, -2, 3, 0, -4];
    let result = prod_signs(&arr);
    println!("Product of signs: {}", result);
}