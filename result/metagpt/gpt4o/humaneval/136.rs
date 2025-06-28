fn largest_smallest_integers(lst: &[i32]) -> (i32, i32) {
    let mut maxneg = i32::MIN;
    let mut minpos = i32::MAX;

    for &value in lst.iter() {
        if value < 0 && value > maxneg {
            maxneg = value;
        }
        if value > 0 && value < minpos {
            minpos = value;
        }
    }

    // If no negative or positive numbers were found, set them to 0
    if maxneg == i32::MIN {
        maxneg = 0;
    }
    if minpos == i32::MAX {
        minpos = 0;
    }

    (maxneg, minpos)
}

fn main() {
    let lst = vec![-10, -20, 5, 15, 0];
    let (maxneg, minpos) = largest_smallest_integers(&lst);
    println!("Largest negative integer: {}", maxneg);
    println!("Smallest positive integer: {}", minpos);
}