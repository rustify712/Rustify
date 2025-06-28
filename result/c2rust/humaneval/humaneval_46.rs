use ::libc;
#[no_mangle]
pub unsafe extern "C" fn fib4(mut n: libc::c_int) -> libc::c_int {
    let mut f: [libc::c_int; 100] = [0; 100];
    f[0 as libc::c_int as usize] = 0 as libc::c_int;
    f[1 as libc::c_int as usize] = 0 as libc::c_int;
    f[2 as libc::c_int as usize] = 2 as libc::c_int;
    f[3 as libc::c_int as usize] = 0 as libc::c_int;
    let mut i: libc::c_int = 4 as libc::c_int;
    while i <= n {
        f[i
            as usize] = f[(i - 1 as libc::c_int) as usize]
            + f[(i - 2 as libc::c_int) as usize] + f[(i - 3 as libc::c_int) as usize]
            + f[(i - 4 as libc::c_int) as usize];
        i += 1;
        i;
    }
    return f[n as usize];
}
