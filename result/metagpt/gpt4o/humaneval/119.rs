fn match_parens(lst: [&str; 2]) -> &'static str {
    let mut l1 = String::new();
    l1.push_str(lst[0]);
    l1.push_str(lst[1]);

    let mut count = 0;
    let mut can = true;

    for ch in l1.chars() {
        if ch == '(' {
            count += 1;
        } else if ch == ')' {
            count -= 1;
        }
        if count < 0 {
            can = false;
        }
    }

    if count != 0 {
        return "No";
    }
    if can {
        return "Yes";
    }

    l1.clear();
    l1.push_str(lst[1]);
    l1.push_str(lst[0]);

    count = 0;
    can = true;

    for ch in l1.chars() {
        if ch == '(' {
            count += 1;
        } else if ch == ')' {
            count -= 1;
        }
        if count < 0 {
            can = false;
        }
    }

    if can {
        "Yes"
    } else {
        "No"
    }
}

fn main() {
    let lst = ["(()", ")()"];
    let result = match_parens(lst);
    println!("Balanced: {}", result);
}