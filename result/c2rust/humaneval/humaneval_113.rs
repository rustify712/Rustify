use ::libc;
extern "C" {
    fn sprintf(_: *mut libc::c_char, _: *const libc::c_char, _: ...) -> libc::c_int;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn strcat(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn odd_count(
    mut lst: *mut *mut libc::c_char,
    mut lst_size: libc::c_int,
) -> *mut *mut libc::c_char {
    let mut out: *mut *mut libc::c_char = malloc(
        (lst_size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<*mut libc::c_char>() as libc::c_ulong),
    ) as *mut *mut libc::c_char;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < lst_size {
        let mut sum: libc::c_int = 0 as libc::c_int;
        let mut j: libc::c_int = 0 as libc::c_int;
        while (j as libc::c_ulong) < strlen(*lst.offset(i as isize)) {
            if *(*lst.offset(i as isize)).offset(j as isize) as libc::c_int >= '0' as i32
                && *(*lst.offset(i as isize)).offset(j as isize) as libc::c_int
                    <= '9' as i32
                && (*(*lst.offset(i as isize)).offset(j as isize) as libc::c_int
                    - '0' as i32) % 2 as libc::c_int == 1 as libc::c_int
            {
                sum += 1 as libc::c_int;
            }
            j += 1;
            j;
        }
        let mut s: *mut libc::c_char = b"the number of odd elements in the string i of the input.\0"
            as *const u8 as *const libc::c_char as *mut libc::c_char;
        let mut s2: *mut libc::c_char = malloc(
            (strlen(s))
                .wrapping_add(20 as libc::c_int as libc::c_ulong)
                .wrapping_mul(::core::mem::size_of::<libc::c_char>() as libc::c_ulong),
        ) as *mut libc::c_char;
        let mut k: libc::c_int = 0 as libc::c_int;
        let mut j_0: libc::c_int = 0 as libc::c_int;
        while (j_0 as libc::c_ulong) < strlen(s) {
            if *s.offset(j_0 as isize) as libc::c_int == 'i' as i32 {
                let mut num_str: [libc::c_char; 20] = [0; 20];
                sprintf(
                    num_str.as_mut_ptr(),
                    b"%d\0" as *const u8 as *const libc::c_char,
                    sum,
                );
                strcat(s2, num_str.as_mut_ptr());
                k = (k as libc::c_ulong).wrapping_add(strlen(num_str.as_mut_ptr()))
                    as libc::c_int as libc::c_int;
            } else {
                let fresh0 = k;
                k = k + 1;
                *s2.offset(fresh0 as isize) = *s.offset(j_0 as isize);
            }
            j_0 += 1;
            j_0;
        }
        *s2.offset(k as isize) = '\0' as i32 as libc::c_char;
        let ref mut fresh1 = *out.offset(i as isize);
        *fresh1 = s2;
        i += 1;
        i;
    }
    return out;
}
