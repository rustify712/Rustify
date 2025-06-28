use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn strncpy(
        _: *mut libc::c_char,
        _: *const libc::c_char,
        _: libc::c_ulong,
    ) -> *mut libc::c_char;
    fn strcat(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn encode_cyclic(mut s: *const libc::c_char) -> *mut libc::c_char {
    let mut l: libc::c_int = strlen(s) as libc::c_int;
    let mut num: libc::c_int = (l + 2 as libc::c_int) / 3 as libc::c_int;
    let mut output: *mut libc::c_char = malloc((l + 1 as libc::c_int) as libc::c_ulong)
        as *mut libc::c_char;
    *output.offset(0 as libc::c_int as isize) = '\0' as i32 as libc::c_char;
    let mut i: libc::c_int = 0 as libc::c_int;
    while (i * 3 as libc::c_int) < l {
        let mut x: [libc::c_char; 4] = [0 as libc::c_int as libc::c_char, 0, 0, 0];
        strncpy(
            x.as_mut_ptr(),
            s.offset((i * 3 as libc::c_int) as isize),
            3 as libc::c_int as libc::c_ulong,
        );
        if strlen(x.as_mut_ptr()) == 3 as libc::c_int as libc::c_ulong {
            let mut temp: libc::c_char = x[0 as libc::c_int as usize];
            x[0 as libc::c_int as usize] = x[1 as libc::c_int as usize];
            x[1 as libc::c_int as usize] = x[2 as libc::c_int as usize];
            x[2 as libc::c_int as usize] = temp;
        }
        strcat(output, x.as_mut_ptr());
        i += 1;
        i;
    }
    return output;
}
#[no_mangle]
pub unsafe extern "C" fn decode_cyclic(mut s: *const libc::c_char) -> *mut libc::c_char {
    let mut l: libc::c_int = strlen(s) as libc::c_int;
    let mut num: libc::c_int = (l + 2 as libc::c_int) / 3 as libc::c_int;
    let mut output: *mut libc::c_char = malloc((l + 1 as libc::c_int) as libc::c_ulong)
        as *mut libc::c_char;
    *output.offset(0 as libc::c_int as isize) = '\0' as i32 as libc::c_char;
    let mut i: libc::c_int = 0 as libc::c_int;
    while (i * 3 as libc::c_int) < l {
        let mut x: [libc::c_char; 4] = [0 as libc::c_int as libc::c_char, 0, 0, 0];
        strncpy(
            x.as_mut_ptr(),
            s.offset((i * 3 as libc::c_int) as isize),
            3 as libc::c_int as libc::c_ulong,
        );
        if strlen(x.as_mut_ptr()) == 3 as libc::c_int as libc::c_ulong {
            let mut temp: libc::c_char = x[2 as libc::c_int as usize];
            x[2 as libc::c_int as usize] = x[1 as libc::c_int as usize];
            x[1 as libc::c_int as usize] = x[0 as libc::c_int as usize];
            x[0 as libc::c_int as usize] = temp;
        }
        strcat(output, x.as_mut_ptr());
        i += 1;
        i;
    }
    return output;
}
