use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn f(mut n: libc::c_int) -> *mut libc::c_int {
    let mut sum: libc::c_int = 0 as libc::c_int;
    let mut prod: libc::c_int = 1 as libc::c_int;
    let mut out: *mut libc::c_int = malloc(
        (n as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    let mut i: libc::c_int = 1 as libc::c_int;
    while i <= n {
        sum += i;
        prod *= i;
        if i % 2 as libc::c_int == 0 as libc::c_int {
            *out.offset((i - 1 as libc::c_int) as isize) = prod;
        } else {
            *out.offset((i - 1 as libc::c_int) as isize) = sum;
        }
        i += 1;
        i;
    }
    return out;
}
