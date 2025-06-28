use ::libc;
#[no_mangle]
pub unsafe extern "C" fn exchange(
    mut lst1: *mut libc::c_int,
    mut lst1_size: libc::c_int,
    mut lst2: *mut libc::c_int,
    mut lst2_size: libc::c_int,
) -> *const libc::c_char {
    let mut num: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < lst1_size {
        if *lst1.offset(i as isize) % 2 as libc::c_int == 0 as libc::c_int {
            num += 1;
            num;
        }
        i += 1;
        i;
    }
    let mut i_0: libc::c_int = 0 as libc::c_int;
    while i_0 < lst2_size {
        if *lst2.offset(i_0 as isize) % 2 as libc::c_int == 0 as libc::c_int {
            num += 1;
            num;
        }
        i_0 += 1;
        i_0;
    }
    if num >= lst1_size {
        return b"YES\0" as *const u8 as *const libc::c_char;
    }
    return b"NO\0" as *const u8 as *const libc::c_char;
}
