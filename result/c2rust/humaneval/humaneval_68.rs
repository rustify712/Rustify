use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn free(_: *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn pluck(
    mut arr: *mut libc::c_int,
    mut size: libc::c_int,
    mut returnSize: *mut libc::c_int,
) -> *mut libc::c_int {
    let mut out: *mut libc::c_int = malloc(
        (2 as libc::c_int as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    *returnSize = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        if *arr.offset(i as isize) % 2 as libc::c_int == 0 as libc::c_int
            && (*returnSize == 0 as libc::c_int
                || *arr.offset(i as isize) < *out.offset(0 as libc::c_int as isize))
        {
            *out.offset(0 as libc::c_int as isize) = *arr.offset(i as isize);
            *out.offset(1 as libc::c_int as isize) = i;
            *returnSize = 2 as libc::c_int;
        }
        i += 1;
        i;
    }
    if *returnSize == 0 as libc::c_int {
        free(out as *mut libc::c_void);
        return 0 as *mut libc::c_int;
    }
    return out;
}
