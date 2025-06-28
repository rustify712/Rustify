pub fn find_closest_elements(numbers: &[f32]) -> (f32, f32) {
    let mut min_diff = f32::INFINITY;
    let mut min_i = 0;
    let mut min_j = 1;

    for i in 0..numbers.len() {
        for j in (i + 1)..numbers.len() {
            let diff = (numbers[i] - numbers[j]).abs();
            if diff < min_diff {
                min_diff = diff;
                min_i = i;
                min_j = j;
            }
        }
    }

    if numbers[min_i] > numbers[min_j] {
        (numbers[min_j], numbers[min_i])
    } else {
        (numbers[min_i], numbers[min_j])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_closest_elements() {
        assert_eq!(
            find_closest_elements(&[1.0, 2.0, 3.0, 4.0, 1.5]),
            (1.0, 1.5)
        );
        assert_eq!(
            find_closest_elements(&[5.0, 3.0, 8.0, 3.2]),
            (3.0, 3.2)
        );
    }
}