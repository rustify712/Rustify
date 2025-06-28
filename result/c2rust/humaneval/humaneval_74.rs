use ::libc;
extern "C" {
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct StringVector {
    pub strings: *mut *mut libc::c_char,
    pub size: libc::c_int,
}
#[no_mangle]
pub unsafe extern "C" fn calculate_total_chars(
    mut vec: *mut StringVector,
) -> libc::c_int {
    let mut total: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < (*vec).size {
        total = (total as libc::c_ulong)
            .wrapping_add(strlen(*((*vec).strings).offset(i as isize))) as libc::c_int
            as libc::c_int;
        i += 1;
        i;
    }
    return total;
}
#[no_mangle]
pub unsafe extern "C" fn total_match(
    mut lst1: StringVector,
    mut lst2: StringVector,
) -> StringVector {
    let mut num1: libc::c_int = calculate_total_chars(&mut lst1);
    let mut num2: libc::c_int = calculate_total_chars(&mut lst2);
    if num1 > num2 {
        return lst2;
    }
    return lst1;
}
