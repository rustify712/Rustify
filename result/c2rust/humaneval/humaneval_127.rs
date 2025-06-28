use ::libc;
#[no_mangle]
pub unsafe extern "C" fn intersection(
    mut interval1: *mut libc::c_int,
    mut interval2: *mut libc::c_int,
) -> *mut libc::c_char {
    let mut inter1: libc::c_int = 0;
    let mut inter2: libc::c_int = 0;
    let mut l: libc::c_int = 0;
    let mut i: libc::c_int = 0;
    inter1 = if *interval1.offset(0 as libc::c_int as isize)
        > *interval2.offset(0 as libc::c_int as isize)
    {
        *interval1.offset(0 as libc::c_int as isize)
    } else {
        *interval2.offset(0 as libc::c_int as isize)
    };
    inter2 = if *interval1.offset(1 as libc::c_int as isize)
        < *interval2.offset(1 as libc::c_int as isize)
    {
        *interval1.offset(1 as libc::c_int as isize)
    } else {
        *interval2.offset(1 as libc::c_int as isize)
    };
    l = inter2 - inter1;
    if l < 2 as libc::c_int {
        return b"NO\0" as *const u8 as *const libc::c_char as *mut libc::c_char;
    }
    i = 2 as libc::c_int;
    while i * i <= l {
        if l % i == 0 as libc::c_int {
            return b"NO\0" as *const u8 as *const libc::c_char as *mut libc::c_char;
        }
        i += 1;
        i;
    }
    return b"YES\0" as *const u8 as *const libc::c_char as *mut libc::c_char;
}
