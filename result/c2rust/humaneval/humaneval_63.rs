use ::libc;
#[no_mangle]
pub unsafe extern "C" fn fibfib(mut n: libc::c_int) -> libc::c_int {
    let mut ff: [libc::c_int; 100] = [0; 100];
    ff[0 as libc::c_int as usize] = 0 as libc::c_int;
    ff[1 as libc::c_int as usize] = 0 as libc::c_int;
    ff[2 as libc::c_int as usize] = 1 as libc::c_int;
    let mut i: libc::c_int = 3 as libc::c_int;
    while i <= n {
        ff[i
            as usize] = ff[(i - 1 as libc::c_int) as usize]
            + ff[(i - 2 as libc::c_int) as usize] + ff[(i - 3 as libc::c_int) as usize];
        i += 1;
        i;
    }
    return ff[n as usize];
}
