/// Sorts a vector of integers based on the sum of their digits.
///
/// # Arguments
///
/// * `nums` - A vector of integers to be sorted.
///
/// # Returns
///
/// A vector of integers sorted by the sum of their digits.
fn order_by_points(nums: Vec<i32>) -> Vec<i32> {
    if nums.is_empty() {
        return vec![];
    }

    // Create a vector of tuples containing (original number, digit sum, original index)
    let mut number_info: Vec<(i32, i32, usize)> = nums.iter()
        .enumerate()
        .map(|(idx, &num)| {
            let num_str = num.abs().to_string();
            let mut sum = 0;

            // Calculate digit sum
            for (i, c) in num_str.chars().enumerate() {
                let digit = (c as i32) - ('0' as i32);
                if i == 0 {
                    // Handle first digit's sign for negative numbers
                    sum += if num < 0 { -digit } else { digit };
                } else {
                    sum += digit;
                }
            }

            (num, sum, idx)
        })
        .collect();

    // Sort by digit sum, then by original index for stable sort
    number_info.sort_by(|a, b| {
        if a.1 == b.1 {
            a.2.cmp(&b.2)  // If sums are equal, preserve original order
        } else {
            a.1.cmp(&b.1)  // Sort by digit sum
        }
    });

    // Extract the sorted numbers
    number_info.into_iter().map(|(num, _, _)| num).collect()
}