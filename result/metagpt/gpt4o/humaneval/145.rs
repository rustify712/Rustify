fn order_by_points(nums: &mut [i32]) -> &mut [i32] {
    let size = nums.len();
    let mut sumdigit = vec![0; size];

    for i in 0..size {
        let w = nums[i].abs().to_string();
        let mut sum = 0;
        for ch in w.chars().skip(1) {
            sum += ch.to_digit(10).unwrap() as i32;
        }
        if nums[i] > 0 {
            sum += w.chars().next().unwrap().to_digit(10).unwrap() as i32;
        } else {
            sum -= w.chars().next().unwrap().to_digit(10).unwrap() as i32;
        }
        sumdigit[i] = sum;
    }

    for i in 0..size {
        for j in 1..size {
            if sumdigit[j - 1] > sumdigit[j] {
                sumdigit.swap(j, j - 1);
                nums.swap(j, j - 1);
            }
        }
    }

    nums
}

fn main() {
    let mut nums = vec![123, -456, 789, -1011, 1213];
    let sorted_nums = order_by_points(&mut nums);
    println!("Sorted by points: {:?}", sorted_nums);
}