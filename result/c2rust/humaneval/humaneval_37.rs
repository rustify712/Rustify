use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn free(_: *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn sort_even(
    mut l: *mut libc::c_float,
    mut size: libc::c_int,
    mut out_size: *mut libc::c_int,
) -> *mut libc::c_float {
    let mut out: *mut libc::c_float = malloc(
        (size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_float>() as libc::c_ulong),
    ) as *mut libc::c_float;
    if out.is_null() {
        *out_size = 0 as libc::c_int;
        return 0 as *mut libc::c_float;
    }
    let mut even_size: libc::c_int = (size + 1 as libc::c_int) / 2 as libc::c_int;
    let mut even: *mut libc::c_float = malloc(
        (even_size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_float>() as libc::c_ulong),
    ) as *mut libc::c_float;
    if even.is_null() {
        free(out as *mut libc::c_void);
        *out_size = 0 as libc::c_int;
        return 0 as *mut libc::c_float;
    }
    let mut i: libc::c_int = 0 as libc::c_int;
    while (i * 2 as libc::c_int) < size {
        *even.offset(i as isize) = *l.offset((i * 2 as libc::c_int) as isize);
        i += 1;
        i;
    }
    let mut i_0: libc::c_int = 0 as libc::c_int;
    while i_0 < even_size - 1 as libc::c_int {
        let mut j: libc::c_int = i_0 + 1 as libc::c_int;
        while j < even_size {
            if *even.offset(i_0 as isize) > *even.offset(j as isize) {
                let mut temp: libc::c_float = *even.offset(i_0 as isize);
                *even.offset(i_0 as isize) = *even.offset(j as isize);
                *even.offset(j as isize) = temp;
            }
            j += 1;
            j;
        }
        i_0 += 1;
        i_0;
    }
    let mut i_1: libc::c_int = 0 as libc::c_int;
    while i_1 < size {
        if i_1 % 2 as libc::c_int == 0 as libc::c_int {
            *out.offset(i_1 as isize) = *even.offset((i_1 / 2 as libc::c_int) as isize);
        } else {
            *out.offset(i_1 as isize) = *l.offset(i_1 as isize);
        }
        i_1 += 1;
        i_1;
    }
    free(even as *mut libc::c_void);
    *out_size = size;
    return out;
}
