use ::libc;
extern "C" {
    fn sprintf(_: *mut libc::c_char, _: *const libc::c_char, _: ...) -> libc::c_int;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn free(_: *mut libc::c_void);
    fn abs(_: libc::c_int) -> libc::c_int;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn order_by_points(
    mut nums: *mut libc::c_int,
    mut size: libc::c_int,
) -> *mut libc::c_int {
    let mut sumdigit: *mut libc::c_int = malloc(
        (size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < size {
        let mut w: [libc::c_char; 20] = [0; 20];
        sprintf(
            w.as_mut_ptr(),
            b"%d\0" as *const u8 as *const libc::c_char,
            abs(*nums.offset(i as isize)),
        );
        let mut sum: libc::c_int = 0 as libc::c_int;
        let mut j: libc::c_int = 1 as libc::c_int;
        while (j as libc::c_ulong) < strlen(w.as_mut_ptr()) {
            sum += w[j as usize] as libc::c_int - '0' as i32;
            j += 1;
            j;
        }
        if *nums.offset(i as isize) > 0 as libc::c_int {
            sum += w[0 as libc::c_int as usize] as libc::c_int - '0' as i32;
        } else {
            sum -= w[0 as libc::c_int as usize] as libc::c_int - '0' as i32;
        }
        *sumdigit.offset(i as isize) = sum;
        i += 1;
        i;
    }
    let mut m: libc::c_int = 0;
    let mut i_0: libc::c_int = 0 as libc::c_int;
    while i_0 < size {
        let mut j_0: libc::c_int = 1 as libc::c_int;
        while j_0 < size {
            if *sumdigit.offset((j_0 - 1 as libc::c_int) as isize)
                > *sumdigit.offset(j_0 as isize)
            {
                m = *sumdigit.offset(j_0 as isize);
                *sumdigit
                    .offset(
                        j_0 as isize,
                    ) = *sumdigit.offset((j_0 - 1 as libc::c_int) as isize);
                *sumdigit.offset((j_0 - 1 as libc::c_int) as isize) = m;
                m = *nums.offset(j_0 as isize);
                *nums
                    .offset(
                        j_0 as isize,
                    ) = *nums.offset((j_0 - 1 as libc::c_int) as isize);
                *nums.offset((j_0 - 1 as libc::c_int) as isize) = m;
            }
            j_0 += 1;
            j_0;
        }
        i_0 += 1;
        i_0;
    }
    free(sumdigit as *mut libc::c_void);
    return nums;
}
