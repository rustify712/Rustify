fn do_algebra(operato: Vec<String>, mut operand: Vec<i32>) -> i32 {
    let mut posto: Vec<usize> = (0..operand.len()).collect();

    // Handle exponentiation first (**)
    let mut i = 0;
    while i < operato.len() {
        if operato[i] == "**" {
            let idx1 = i;
            let idx2 = i + 1;

            let root1 = find_root(&mut posto, idx1);
            let root2 = find_root(&mut posto, idx2);

            let base = operand[root1] as i32;
            let exp = operand[root2] as i32;
            operand[root1] = base.pow(exp as u32);
            posto[root2] = root1;
        }
        i += 1;
    }

    // Handle multiplication and division
    let mut i = 0;
    while i < operato.len() {
        if operato[i] == "*" || operato[i] == "//" {
            let idx1 = i;
            let idx2 = i + 1;

            let root1 = find_root(&mut posto, idx1);
            let root2 = find_root(&mut posto, idx2);

            operand[root1] = if operato[i] == "*" {
                operand[root1] * operand[root2]
            } else {
                operand[root1] / operand[root2]
            };
            posto[root2] = root1;
        }
        i += 1;
    }

    // Handle addition and subtraction
    let mut i = 0;
    while i < operato.len() {
        if operato[i] == "+" || operato[i] == "-" {
            let idx1 = i;
            let idx2 = i + 1;

            let root1 = find_root(&mut posto, idx1);
            let root2 = find_root(&mut posto, idx2);

            operand[root1] = if operato[i] == "+" {
                operand[root1] + operand[root2]
            } else {
                operand[root1] - operand[root2]
            };
            posto[root2] = root1;
        }
        i += 1;
    }

    operand[find_root(&mut posto, 0)]
}

fn find_root(posto: &mut Vec<usize>, mut idx: usize) -> usize {
    while posto[idx] != idx {
        let parent = posto[idx];
        posto[idx] = posto[parent];
        idx = posto[idx];
    }
    idx
}