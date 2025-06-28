use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn unique_digits(
    mut x: *mut libc::c_int,
    mut size: libc::c_int,
    mut result_size: *mut libc::c_int,
) -> *mut libc::c_int {
    let mut out: *mut libc::c_int = malloc(
        (size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    let mut out_index: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        let mut num: libc::c_int = *x.offset(i as isize);
        let mut u: libc::c_int = 1 as libc::c_int;
        if num == 0 as libc::c_int {
            u = 0 as libc::c_int;
        }
        while num > 0 as libc::c_int && u != 0 {
            if num % 2 as libc::c_int == 0 as libc::c_int {
                u = 0 as libc::c_int;
            }
            num = num / 10 as libc::c_int;
        }
        if u != 0 {
            let fresh0 = out_index;
            out_index = out_index + 1;
            *out.offset(fresh0 as isize) = *x.offset(i as isize);
        }
        i += 1;
        i;
    }
    let mut i_0: libc::c_int = 0 as libc::c_int;
    while i_0 < out_index - 1 as libc::c_int {
        let mut j: libc::c_int = i_0 + 1 as libc::c_int;
        while j < out_index {
            if *out.offset(i_0 as isize) > *out.offset(j as isize) {
                let mut temp: libc::c_int = *out.offset(i_0 as isize);
                *out.offset(i_0 as isize) = *out.offset(j as isize);
                *out.offset(j as isize) = temp;
            }
            j += 1;
            j;
        }
        i_0 += 1;
        i_0;
    }
    *result_size = out_index;
    return out;
}
