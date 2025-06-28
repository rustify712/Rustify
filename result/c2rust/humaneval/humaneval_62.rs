use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn derivative(
    mut xs: *mut libc::c_float,
    mut size: libc::c_int,
    mut out_size: *mut libc::c_int,
) -> *mut libc::c_float {
    *out_size = size - 1 as libc::c_int;
    let mut out: *mut libc::c_float = malloc(
        (*out_size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_float>() as libc::c_ulong),
    ) as *mut libc::c_float;
    let mut i: libc::c_int = 1 as libc::c_int;
    while i < size {
        *out
            .offset(
                (i - 1 as libc::c_int) as isize,
            ) = i as libc::c_float * *xs.offset(i as isize);
        i += 1;
        i;
    }
    return out;
}
