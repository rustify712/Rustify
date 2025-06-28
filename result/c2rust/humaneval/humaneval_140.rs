use ::libc;
extern "C" {
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn fix_spaces(mut text: *const libc::c_char) -> *mut libc::c_char {
    let mut len: libc::c_int = strlen(text) as libc::c_int;
    let mut out: *mut libc::c_char = malloc(
        ((2 as libc::c_int * len) as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_char>() as libc::c_ulong),
    ) as *mut libc::c_char;
    if out.is_null() {
        return 0 as *mut libc::c_char;
    }
    let mut out_index: libc::c_int = 0 as libc::c_int;
    let mut spacelen: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < len {
        if *text.offset(i as isize) as libc::c_int == ' ' as i32 {
            spacelen += 1;
            spacelen;
        } else {
            if spacelen == 1 as libc::c_int {
                let fresh0 = out_index;
                out_index = out_index + 1;
                *out.offset(fresh0 as isize) = '_' as i32 as libc::c_char;
            } else if spacelen == 2 as libc::c_int {
                let fresh1 = out_index;
                out_index = out_index + 1;
                *out.offset(fresh1 as isize) = '_' as i32 as libc::c_char;
                let fresh2 = out_index;
                out_index = out_index + 1;
                *out.offset(fresh2 as isize) = '_' as i32 as libc::c_char;
            } else if spacelen > 2 as libc::c_int {
                let fresh3 = out_index;
                out_index = out_index + 1;
                *out.offset(fresh3 as isize) = '-' as i32 as libc::c_char;
            }
            spacelen = 0 as libc::c_int;
            let fresh4 = out_index;
            out_index = out_index + 1;
            *out.offset(fresh4 as isize) = *text.offset(i as isize);
        }
        i += 1;
        i;
    }
    if spacelen == 1 as libc::c_int {
        let fresh5 = out_index;
        out_index = out_index + 1;
        *out.offset(fresh5 as isize) = '_' as i32 as libc::c_char;
    } else if spacelen == 2 as libc::c_int {
        let fresh6 = out_index;
        out_index = out_index + 1;
        *out.offset(fresh6 as isize) = '_' as i32 as libc::c_char;
        let fresh7 = out_index;
        out_index = out_index + 1;
        *out.offset(fresh7 as isize) = '_' as i32 as libc::c_char;
    } else if spacelen > 2 as libc::c_int {
        let fresh8 = out_index;
        out_index = out_index + 1;
        *out.offset(fresh8 as isize) = '-' as i32 as libc::c_char;
    }
    *out.offset(out_index as isize) = '\0' as i32 as libc::c_char;
    return out;
}
