use ::libc;
#[no_mangle]
pub unsafe extern "C" fn triples_sum_to_zero(
    mut l: *mut libc::c_int,
    mut size: libc::c_int,
) -> bool {
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        let mut j: libc::c_int = i + 1 as libc::c_int;
        while j < size {
            let mut k: libc::c_int = j + 1 as libc::c_int;
            while k < size {
                if *l.offset(i as isize) + *l.offset(j as isize) + *l.offset(k as isize)
                    == 0 as libc::c_int
                {
                    return 1 as libc::c_int != 0;
                }
                k += 1;
                k;
            }
            j += 1;
            j;
        }
        i += 1;
        i;
    }
    return 0 as libc::c_int != 0;
}
