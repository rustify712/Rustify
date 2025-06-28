use ::libc;
extern "C" {
    fn sprintf(_: *mut libc::c_char, _: *const libc::c_char, _: ...) -> libc::c_int;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn even_odd_palindrome(
    mut n: libc::c_int,
    mut even_count: *mut libc::c_int,
    mut odd_count: *mut libc::c_int,
) {
    *even_count = 0 as libc::c_int;
    *odd_count = 0 as libc::c_int;
    let mut i: libc::c_int = 1 as libc::c_int;
    while i <= n {
        let mut w: [libc::c_char; 12] = [0; 12];
        sprintf(w.as_mut_ptr(), b"%d\0" as *const u8 as *const libc::c_char, i);
        let mut len: libc::c_int = strlen(w.as_mut_ptr()) as libc::c_int;
        let mut is_palindrome: libc::c_int = 1 as libc::c_int;
        let mut j: libc::c_int = 0 as libc::c_int;
        while j < len / 2 as libc::c_int {
            if w[j as usize] as libc::c_int
                != w[(len - j - 1 as libc::c_int) as usize] as libc::c_int
            {
                is_palindrome = 0 as libc::c_int;
                break;
            } else {
                j += 1;
                j;
            }
        }
        if is_palindrome != 0 {
            if i % 2 as libc::c_int == 0 as libc::c_int {
                *even_count += 1;
                *even_count;
            } else {
                *odd_count += 1;
                *odd_count;
            }
        }
        i += 1;
        i;
    }
}
