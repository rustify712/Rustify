use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn rolling_max(
    mut numbers: *mut libc::c_int,
    mut size: libc::c_int,
    mut out_size: *mut libc::c_int,
) -> *mut libc::c_int {
    let mut out: *mut libc::c_int = malloc(
        (size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    if out.is_null() {
        *out_size = 0 as libc::c_int;
        return 0 as *mut libc::c_int;
    }
    let mut max: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        if *numbers.offset(i as isize) > max {
            max = *numbers.offset(i as isize);
        }
        *out.offset(i as isize) = max;
        i += 1;
        i;
    }
    *out_size = size;
    return out;
}
