pub fn strange_sort_list(mut lst: Vec<i32>) -> Vec<i32> {
    lst.sort_unstable();
    let mut result = Vec::with_capacity(lst.len());
    let (mut l, mut r) = (0, lst.len() - 1);
    
    while l < r {
        result.push(lst[l]);
        result.push(lst[r]);
        l += 1;
        r -= 1;
    }
    
    if l == r {
        result.push(lst[l]);
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strange_sort_list() {
        assert_eq!(strange_sort_list(vec![1, 2, 3, 4]), vec![1, 4, 2, 3]);
        assert_eq!(strange_sort_list(vec![5, 2, 3, 1]), vec![1, 5, 2, 3]);
        assert_eq!(strange_sort_list(vec![1, 2, 3]), vec![1, 3, 2]);
    }
}