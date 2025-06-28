use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn count_up_to(
    mut n: libc::c_int,
    mut returnSize: *mut libc::c_int,
) -> *mut libc::c_int {
    let mut out: *mut libc::c_int = malloc(
        (n as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    let mut count: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0;
    let mut j: libc::c_int = 0;
    i = 2 as libc::c_int;
    while i < n {
        if count == 0 as libc::c_int {
            let fresh0 = count;
            count = count + 1;
            *out.offset(fresh0 as isize) = i;
        } else {
            let mut isp: libc::c_int = 1 as libc::c_int;
            j = 0 as libc::c_int;
            while *out.offset(j as isize) * *out.offset(j as isize) <= i {
                if i % *out.offset(j as isize) == 0 as libc::c_int {
                    isp = 0 as libc::c_int;
                    break;
                } else {
                    j += 1;
                    j;
                }
            }
            if isp != 0 {
                let fresh1 = count;
                count = count + 1;
                *out.offset(fresh1 as isize) = i;
            }
        }
        i += 1;
        i;
    }
    *returnSize = count;
    return out;
}
