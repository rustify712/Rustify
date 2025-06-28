use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn free(_: *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn sort_third(
    mut l: *mut libc::c_int,
    mut size: libc::c_int,
) -> *mut libc::c_int {
    let mut third: *mut libc::c_int = malloc(
        (size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    let mut third_size: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while (i * 3 as libc::c_int) < size {
        let fresh0 = third_size;
        third_size = third_size + 1;
        *third.offset(fresh0 as isize) = *l.offset((i * 3 as libc::c_int) as isize);
        i += 1;
        i;
    }
    let mut i_0: libc::c_int = 0 as libc::c_int;
    while i_0 < third_size - 1 as libc::c_int {
        let mut j: libc::c_int = i_0 + 1 as libc::c_int;
        while j < third_size {
            if *third.offset(i_0 as isize) > *third.offset(j as isize) {
                let mut temp: libc::c_int = *third.offset(i_0 as isize);
                *third.offset(i_0 as isize) = *third.offset(j as isize);
                *third.offset(j as isize) = temp;
            }
            j += 1;
            j;
        }
        i_0 += 1;
        i_0;
    }
    let mut out: *mut libc::c_int = malloc(
        (size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    let mut i_1: libc::c_int = 0 as libc::c_int;
    while i_1 < size {
        if i_1 % 3 as libc::c_int == 0 as libc::c_int {
            *out.offset(i_1 as isize) = *third.offset((i_1 / 3 as libc::c_int) as isize);
        } else {
            *out.offset(i_1 as isize) = *l.offset(i_1 as isize);
        }
        i_1 += 1;
        i_1;
    }
    free(third as *mut libc::c_void);
    return out;
}
