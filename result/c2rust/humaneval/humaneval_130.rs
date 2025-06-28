use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn tri(
    mut n: libc::c_int,
    mut returnSize: *mut libc::c_int,
) -> *mut libc::c_int {
    *returnSize = n + 1 as libc::c_int;
    let mut out: *mut libc::c_int = malloc(
        ((n + 1 as libc::c_int) as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    if out.is_null() {
        *returnSize = 0 as libc::c_int;
        return 0 as *mut libc::c_int;
    }
    *out.offset(0 as libc::c_int as isize) = 1 as libc::c_int;
    if n == 0 as libc::c_int {
        return out;
    }
    *out.offset(1 as libc::c_int as isize) = 3 as libc::c_int;
    let mut i: libc::c_int = 2 as libc::c_int;
    while i <= n {
        if i % 2 as libc::c_int == 0 as libc::c_int {
            *out.offset(i as isize) = 1 as libc::c_int + i / 2 as libc::c_int;
        } else {
            *out
                .offset(
                    i as isize,
                ) = *out.offset((i - 1 as libc::c_int) as isize)
                + *out.offset((i - 2 as libc::c_int) as isize) + 1 as libc::c_int
                + (i + 1 as libc::c_int) / 2 as libc::c_int;
        }
        i += 1;
        i;
    }
    return out;
}
