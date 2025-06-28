fn min_sub_array_sum(nums: &[i64]) -> i64 {
    let mut current = nums[0];
    let mut min = nums[0];
    for &num in nums.iter().skip(1) {
        if current < 0 {
            current += num;
        } else {
            current = num;
        }
        if current < min {
            min = current;
        }
    }
    min
}

fn main() {
    let nums = vec![3, -4, 2, -3, -1, 7, -5];
    let result = min_sub_array_sum(&nums);
    println!("Minimum subarray sum: {}", result);
}