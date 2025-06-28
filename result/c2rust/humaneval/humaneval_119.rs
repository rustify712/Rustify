use ::libc;
extern "C" {
    fn strcpy(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strcat(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn match_parens(
    mut lst: *mut *mut libc::c_char,
) -> *mut libc::c_char {
    let mut l1: [libc::c_char; 1000] = [0; 1000];
    let mut i: libc::c_int = 0;
    let mut count: libc::c_int = 0 as libc::c_int;
    let mut can: libc::c_int = 1 as libc::c_int;
    strcpy(l1.as_mut_ptr(), *lst.offset(0 as libc::c_int as isize));
    strcat(l1.as_mut_ptr(), *lst.offset(1 as libc::c_int as isize));
    i = 0 as libc::c_int;
    while (i as libc::c_ulong) < strlen(l1.as_mut_ptr()) {
        if l1[i as usize] as libc::c_int == '(' as i32 {
            count += 1 as libc::c_int;
        }
        if l1[i as usize] as libc::c_int == ')' as i32 {
            count -= 1 as libc::c_int;
        }
        if count < 0 as libc::c_int {
            can = 0 as libc::c_int;
        }
        i += 1;
        i;
    }
    if count != 0 as libc::c_int {
        return b"No\0" as *const u8 as *const libc::c_char as *mut libc::c_char;
    }
    if can == 1 as libc::c_int {
        return b"Yes\0" as *const u8 as *const libc::c_char as *mut libc::c_char;
    }
    strcpy(l1.as_mut_ptr(), *lst.offset(1 as libc::c_int as isize));
    strcat(l1.as_mut_ptr(), *lst.offset(0 as libc::c_int as isize));
    count = 0 as libc::c_int;
    can = 1 as libc::c_int;
    i = 0 as libc::c_int;
    while (i as libc::c_ulong) < strlen(l1.as_mut_ptr()) {
        if l1[i as usize] as libc::c_int == '(' as i32 {
            count += 1 as libc::c_int;
        }
        if l1[i as usize] as libc::c_int == ')' as i32 {
            count -= 1 as libc::c_int;
        }
        if count < 0 as libc::c_int {
            can = 0 as libc::c_int;
        }
        i += 1;
        i;
    }
    if can == 1 as libc::c_int {
        return b"Yes\0" as *const u8 as *const libc::c_char as *mut libc::c_char;
    }
    return b"No\0" as *const u8 as *const libc::c_char as *mut libc::c_char;
}
