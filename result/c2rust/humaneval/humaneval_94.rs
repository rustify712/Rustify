use ::libc;
extern "C" {
    fn sprintf(_: *mut libc::c_char, _: *const libc::c_char, _: ...) -> libc::c_int;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn skjkasdkd(
    mut lst: *mut libc::c_int,
    mut size: libc::c_int,
) -> libc::c_int {
    let mut largest: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        if *lst.offset(i as isize) > largest {
            let mut prime: bool = 1 as libc::c_int != 0;
            let mut j: libc::c_int = 2 as libc::c_int;
            while j * j <= *lst.offset(i as isize) {
                if *lst.offset(i as isize) % j == 0 as libc::c_int {
                    prime = 0 as libc::c_int != 0;
                    break;
                } else {
                    j += 1;
                    j;
                }
            }
            if prime {
                largest = *lst.offset(i as isize);
            }
        }
        i += 1;
        i;
    }
    let mut sum: libc::c_int = 0 as libc::c_int;
    let mut s: [libc::c_char; 20] = [0; 20];
    sprintf(s.as_mut_ptr(), b"%d\0" as *const u8 as *const libc::c_char, largest);
    let mut i_0: libc::c_int = 0 as libc::c_int;
    while (i_0 as libc::c_ulong) < strlen(s.as_mut_ptr()) {
        sum += s[i_0 as usize] as libc::c_int - '0' as i32;
        i_0 += 1;
        i_0;
    }
    return sum;
}
