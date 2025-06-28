use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn free(_: *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn search(
    mut lst: *mut libc::c_int,
    mut lst_size: libc::c_int,
) -> libc::c_int {
    let mut max: libc::c_int = -(1 as libc::c_int);
    let mut freq_values: *mut libc::c_int = malloc(
        (lst_size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    let mut freq_counts: *mut libc::c_int = malloc(
        (lst_size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    let mut freq_size: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < lst_size {
        let mut has: bool = 0 as libc::c_int != 0;
        let mut j: libc::c_int = 0 as libc::c_int;
        while j < freq_size {
            if *lst.offset(i as isize) == *freq_values.offset(j as isize) {
                let ref mut fresh0 = *freq_counts.offset(j as isize);
                *fresh0 += 1;
                *fresh0;
                has = 1 as libc::c_int != 0;
                if *freq_counts.offset(j as isize) >= *freq_values.offset(j as isize)
                    && *freq_values.offset(j as isize) > max
                {
                    max = *freq_values.offset(j as isize);
                }
                break;
            } else {
                j += 1;
                j;
            }
        }
        if !has {
            *freq_values.offset(freq_size as isize) = *lst.offset(i as isize);
            *freq_counts.offset(freq_size as isize) = 1 as libc::c_int;
            if max == -(1 as libc::c_int) && *lst.offset(i as isize) == 1 as libc::c_int
            {
                max = 1 as libc::c_int;
            }
            freq_size += 1;
            freq_size;
        }
        i += 1;
        i;
    }
    free(freq_values as *mut libc::c_void);
    free(freq_counts as *mut libc::c_void);
    return max;
}
