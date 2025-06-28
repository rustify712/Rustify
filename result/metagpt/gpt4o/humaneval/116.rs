fn sort_array(arr: &mut [i32]) -> &mut [i32] {
    let size = arr.len();
    let mut bin = vec![0; size];

    for i in 0..size {
        let mut b = 0;
        let mut n = arr[i].abs();
        while n > 0 {
            b += n % 2;
            n /= 2;
        }
        bin[i] = b;
    }

    for i in 0..size {
        for j in 1..size {
            if bin[j] < bin[j - 1] || (bin[j] == bin[j - 1] && arr[j] < arr[j - 1]) {
                arr.swap(j, j - 1);
                bin.swap(j, j - 1);
            }
        }
    }

    arr
}

fn main() {
    let mut arr = vec![3, 8, 3, 6, 5];
    let sorted_arr = sort_array(&mut arr);
    println!("Sorted array: {:?}", sorted_arr);
}