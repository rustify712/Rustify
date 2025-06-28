use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn intersperse(
    mut numbers: *mut libc::c_int,
    mut size: libc::c_int,
    mut delimeter: libc::c_int,
    mut out_size: *mut libc::c_int,
) -> *mut libc::c_int {
    *out_size = if size == 0 as libc::c_int {
        0 as libc::c_int
    } else {
        2 as libc::c_int * size - 1 as libc::c_int
    };
    let mut out: *mut libc::c_int = malloc(
        (*out_size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    if size > 0 as libc::c_int {
        *out
            .offset(
                0 as libc::c_int as isize,
            ) = *numbers.offset(0 as libc::c_int as isize);
        let mut i: libc::c_int = 1 as libc::c_int;
        while i < size {
            *out.offset((2 as libc::c_int * i - 1 as libc::c_int) as isize) = delimeter;
            *out.offset((2 as libc::c_int * i) as isize) = *numbers.offset(i as isize);
            i += 1;
            i;
        }
    }
    return out;
}
