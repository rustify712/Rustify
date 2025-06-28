use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn factorize(
    mut n: libc::c_int,
    mut returnSize: *mut libc::c_int,
) -> *mut libc::c_int {
    let mut out: *mut libc::c_int = malloc(
        (::core::mem::size_of::<libc::c_int>() as libc::c_ulong)
            .wrapping_mul(n as libc::c_ulong),
    ) as *mut libc::c_int;
    let mut count: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 2 as libc::c_int;
    while i * i <= n {
        while n % i == 0 as libc::c_int {
            let fresh0 = count;
            count = count + 1;
            *out.offset(fresh0 as isize) = i;
            n /= i;
        }
        i += 1;
        i;
    }
    if n > 1 as libc::c_int {
        let fresh1 = count;
        count = count + 1;
        *out.offset(fresh1 as isize) = n;
    }
    *returnSize = count;
    return out;
}
