pub fn do_algebra(operators: &[&str], operands: &mut [i32]) -> i32 {
    let mut posto: Vec<usize> = (0..operands.len()).collect();

    // 处理指数运算
    for i in 0..operators.len() {
        if operators[i] == "**" {
            let left = find_root(&mut posto, i);
            let right = find_root(&mut posto, i + 1);
            operands[left] = operands[left].pow(operands[right] as u32);
            posto[right] = left;
        }
    }

    // 处理乘除运算
    for i in 0..operators.len() {
        match operators[i] {
            "*" | "//" => {
                let left = find_root(&mut posto, i);
                let right = find_root(&mut posto, i + 1);
                if operators[i] == "*" {
                    operands[left] *= operands[right];
                } else {
                    operands[left] /= operands[right];
                }
                posto[right] = left;
            }
            _ => {}
        }
    }

    // 处理加减运算
    for i in 0..operators.len() {
        match operators[i] {
            "+" | "-" => {
                let left = find_root(&mut posto, i);
                let right = find_root(&mut posto, i + 1);
                if operators[i] == "+" {
                    operands[left] += operands[right];
                } else {
                    operands[left] -= operands[right];
                }
                posto[right] = left;
            }
            _ => {}
        }
    }

    operands[0]
}

fn find_root(posto: &mut [usize], mut i: usize) -> usize {
    while posto[i] != i {
        posto[i] = posto[posto[i]];
        i = posto[i];
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_do_algebra() {
        // 测试基本运算
        let mut operands = vec![2, 3, 4];
        let operators = vec!["+", "*"];
        assert_eq!(do_algebra(&operators, &mut operands), 14);

        // 测试指数运算
        let mut operands = vec![2, 3];
        let operators = vec!["**"];
        assert_eq!(do_algebra(&operators, &mut operands), 8);

        // 测试混合运算
        let mut operands = vec![2, 3, 4, 5];
        let operators = vec!["**", "*", "+"];
        assert_eq!(do_algebra(&operators, &mut operands), 49);

        // 测试除法运算
        let mut operands = vec![10, 2];
        let operators = vec!["//"];
        assert_eq!(do_algebra(&operators, &mut operands), 5);
    }
}