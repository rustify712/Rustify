use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn generate_integers(
    mut a: libc::c_int,
    mut b: libc::c_int,
    mut returnSize: *mut libc::c_int,
) -> *mut libc::c_int {
    let mut m: libc::c_int = 0;
    if b < a {
        m = a;
        a = b;
        b = m;
    }
    let mut out: *mut libc::c_int = malloc(
        ((b - a + 1 as libc::c_int) as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    let mut count: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = a;
    while i <= b {
        if i < 10 as libc::c_int && i % 2 as libc::c_int == 0 as libc::c_int {
            let fresh0 = count;
            count = count + 1;
            *out.offset(fresh0 as isize) = i;
        }
        i += 1;
        i;
    }
    *returnSize = count;
    return out;
}
