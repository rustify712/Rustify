use ::libc;
#[no_mangle]
pub unsafe extern "C" fn fizz_buzz(mut n: libc::c_int) -> libc::c_int {
    let mut count: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < n {
        if i % 11 as libc::c_int == 0 as libc::c_int
            || i % 13 as libc::c_int == 0 as libc::c_int
        {
            let mut q: libc::c_int = i;
            while q > 0 as libc::c_int {
                if q % 10 as libc::c_int == 7 as libc::c_int {
                    count += 1 as libc::c_int;
                }
                q = q / 10 as libc::c_int;
            }
        }
        i += 1;
        i;
    }
    return count;
}
