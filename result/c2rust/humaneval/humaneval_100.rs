use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn make_a_pile(
    mut n: libc::c_int,
    mut returnSize: *mut libc::c_int,
) -> *mut libc::c_int {
    let mut out: *mut libc::c_int = malloc(
        (n as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    *out.offset(0 as libc::c_int as isize) = n;
    let mut i: libc::c_int = 1 as libc::c_int;
    while i < n {
        *out
            .offset(
                i as isize,
            ) = *out.offset((i - 1 as libc::c_int) as isize) + 2 as libc::c_int;
        i += 1;
        i;
    }
    *returnSize = n;
    return out;
}
