use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn free(_: *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn remove_duplicates(
    mut numbers: *mut libc::c_int,
    mut size: libc::c_int,
    mut result_size: *mut libc::c_int,
) -> *mut libc::c_int {
    let mut out: *mut libc::c_int = malloc(
        (size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    let mut has1: *mut libc::c_int = malloc(
        (size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    let mut has2: *mut libc::c_int = malloc(
        (size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    let mut out_count: libc::c_int = 0 as libc::c_int;
    let mut has1_count: libc::c_int = 0 as libc::c_int;
    let mut has2_count: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        let mut found_in_has2: libc::c_int = 0 as libc::c_int;
        let mut j: libc::c_int = 0 as libc::c_int;
        while j < has2_count {
            if *has2.offset(j as isize) == *numbers.offset(i as isize) {
                found_in_has2 = 1 as libc::c_int;
                break;
            } else {
                j += 1;
                j;
            }
        }
        if !(found_in_has2 != 0) {
            let mut found_in_has1: libc::c_int = 0 as libc::c_int;
            let mut j_0: libc::c_int = 0 as libc::c_int;
            while j_0 < has1_count {
                if *has1.offset(j_0 as isize) == *numbers.offset(i as isize) {
                    found_in_has1 = 1 as libc::c_int;
                    break;
                } else {
                    j_0 += 1;
                    j_0;
                }
            }
            if found_in_has1 != 0 {
                let fresh0 = has2_count;
                has2_count = has2_count + 1;
                *has2.offset(fresh0 as isize) = *numbers.offset(i as isize);
            } else {
                let fresh1 = has1_count;
                has1_count = has1_count + 1;
                *has1.offset(fresh1 as isize) = *numbers.offset(i as isize);
            }
        }
        i += 1;
        i;
    }
    let mut i_0: libc::c_int = 0 as libc::c_int;
    while i_0 < size {
        let mut found_in_has2_0: libc::c_int = 0 as libc::c_int;
        let mut j_1: libc::c_int = 0 as libc::c_int;
        while j_1 < has2_count {
            if *has2.offset(j_1 as isize) == *numbers.offset(i_0 as isize) {
                found_in_has2_0 = 1 as libc::c_int;
                break;
            } else {
                j_1 += 1;
                j_1;
            }
        }
        if found_in_has2_0 == 0 {
            let fresh2 = out_count;
            out_count = out_count + 1;
            *out.offset(fresh2 as isize) = *numbers.offset(i_0 as isize);
        }
        i_0 += 1;
        i_0;
    }
    *result_size = out_count;
    free(has1 as *mut libc::c_void);
    free(has2 as *mut libc::c_void);
    return out;
}
