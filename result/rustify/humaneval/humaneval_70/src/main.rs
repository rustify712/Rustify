/// Sorts the input list and rearranges it by alternating the smallest and largest elements.
///
/// # Arguments
///
/// * `lst` - A slice of integers to be sorted and rearranged.
///
/// # Returns
///
/// A new `Vec<i32>` containing the rearranged elements.
fn strange_sort_list(lst: &[i32]) -> Vec<i32> {
    let mut sorted_lst = lst.to_vec();
    sorted_lst.sort();
    let mut out = Vec::new();
    let mut l = 0;
    let mut r = sorted_lst.len() - 1;
    while l < r {
        out.push(sorted_lst[l]);
        l += 1;
        out.push(sorted_lst[r]);
        r -= 1;
    }
    if l == r {
        out.push(sorted_lst[l]);
    }
    out
}