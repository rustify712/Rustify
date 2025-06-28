use ::libc;
extern "C" {
    fn strcpy(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn free(_: *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn sort_string(mut str: *mut libc::c_char) {
    let mut n: libc::c_int = strlen(str) as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < n - 1 as libc::c_int {
        let mut j: libc::c_int = i + 1 as libc::c_int;
        while j < n {
            if *str.offset(i as isize) as libc::c_int
                > *str.offset(j as isize) as libc::c_int
            {
                let mut temp: libc::c_char = *str.offset(i as isize);
                *str.offset(i as isize) = *str.offset(j as isize);
                *str.offset(j as isize) = temp;
            }
            j += 1;
            j;
        }
        i += 1;
        i;
    }
}
#[no_mangle]
pub unsafe extern "C" fn anti_shuffle(mut s: *const libc::c_char) -> *mut libc::c_char {
    let mut len: libc::c_int = strlen(s) as libc::c_int;
    let mut out: *mut libc::c_char = malloc(
        ((len + 1 as libc::c_int) as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_char>() as libc::c_ulong),
    ) as *mut libc::c_char;
    let mut current: *mut libc::c_char = malloc(
        ((len + 1 as libc::c_int) as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_char>() as libc::c_ulong),
    ) as *mut libc::c_char;
    let mut out_index: libc::c_int = 0 as libc::c_int;
    let mut current_index: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i <= len {
        if *s.offset(i as isize) as libc::c_int == ' ' as i32
            || *s.offset(i as isize) as libc::c_int == '\0' as i32
        {
            *current.offset(current_index as isize) = '\0' as i32 as libc::c_char;
            sort_string(current);
            if out_index > 0 as libc::c_int {
                let fresh0 = out_index;
                out_index = out_index + 1;
                *out.offset(fresh0 as isize) = ' ' as i32 as libc::c_char;
            }
            strcpy(&mut *out.offset(out_index as isize), current);
            out_index = (out_index as libc::c_ulong).wrapping_add(strlen(current))
                as libc::c_int as libc::c_int;
            current_index = 0 as libc::c_int;
        } else {
            let fresh1 = current_index;
            current_index = current_index + 1;
            *current.offset(fresh1 as isize) = *s.offset(i as isize);
        }
        i += 1;
        i;
    }
    *out.offset(out_index as isize) = '\0' as i32 as libc::c_char;
    free(current as *mut libc::c_void);
    return out;
}
