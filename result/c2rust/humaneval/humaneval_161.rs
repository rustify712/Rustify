use ::libc;
extern "C" {
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
    fn __ctype_b_loc() -> *mut *const libc::c_ushort;
    fn __ctype_tolower_loc() -> *mut *const __int32_t;
    fn __ctype_toupper_loc() -> *mut *const __int32_t;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
}
pub type __int32_t = libc::c_int;
pub type C2RustUnnamed = libc::c_uint;
pub const _ISalnum: C2RustUnnamed = 8;
pub const _ISpunct: C2RustUnnamed = 4;
pub const _IScntrl: C2RustUnnamed = 2;
pub const _ISblank: C2RustUnnamed = 1;
pub const _ISgraph: C2RustUnnamed = 32768;
pub const _ISprint: C2RustUnnamed = 16384;
pub const _ISspace: C2RustUnnamed = 8192;
pub const _ISxdigit: C2RustUnnamed = 4096;
pub const _ISdigit: C2RustUnnamed = 2048;
pub const _ISalpha: C2RustUnnamed = 1024;
pub const _ISlower: C2RustUnnamed = 512;
pub const _ISupper: C2RustUnnamed = 256;
#[inline]
unsafe extern "C" fn tolower(mut __c: libc::c_int) -> libc::c_int {
    return if __c >= -(128 as libc::c_int) && __c < 256 as libc::c_int {
        *(*__ctype_tolower_loc()).offset(__c as isize)
    } else {
        __c
    };
}
#[inline]
unsafe extern "C" fn toupper(mut __c: libc::c_int) -> libc::c_int {
    return if __c >= -(128 as libc::c_int) && __c < 256 as libc::c_int {
        *(*__ctype_toupper_loc()).offset(__c as isize)
    } else {
        __c
    };
}
#[no_mangle]
pub unsafe extern "C" fn solve1(mut s: *const libc::c_char) -> *mut libc::c_char {
    let mut nletter: libc::c_int = 0 as libc::c_int;
    let mut len: libc::c_int = strlen(s) as libc::c_int;
    let mut out: *mut libc::c_char = malloc((len + 1 as libc::c_int) as libc::c_ulong)
        as *mut libc::c_char;
    if out.is_null() {
        return 0 as *mut libc::c_char;
    }
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < len {
        let mut w: libc::c_char = *s.offset(i as isize);
        if *(*__ctype_b_loc()).offset(w as libc::c_int as isize) as libc::c_int
            & _ISupper as libc::c_int as libc::c_ushort as libc::c_int != 0
        {
            w = ({
                let mut __res: libc::c_int = 0;
                if ::core::mem::size_of::<libc::c_char>() as libc::c_ulong
                    > 1 as libc::c_int as libc::c_ulong
                {
                    if 0 != 0 {
                        let mut __c: libc::c_int = w as libc::c_int;
                        __res = if __c < -(128 as libc::c_int)
                            || __c > 255 as libc::c_int
                        {
                            __c
                        } else {
                            *(*__ctype_tolower_loc()).offset(__c as isize)
                        };
                    } else {
                        __res = tolower(w as libc::c_int);
                    }
                } else {
                    __res = *(*__ctype_tolower_loc()).offset(w as libc::c_int as isize);
                }
                __res
            }) as libc::c_char;
        } else if *(*__ctype_b_loc()).offset(w as libc::c_int as isize) as libc::c_int
            & _ISlower as libc::c_int as libc::c_ushort as libc::c_int != 0
        {
            w = ({
                let mut __res: libc::c_int = 0;
                if ::core::mem::size_of::<libc::c_char>() as libc::c_ulong
                    > 1 as libc::c_int as libc::c_ulong
                {
                    if 0 != 0 {
                        let mut __c: libc::c_int = w as libc::c_int;
                        __res = if __c < -(128 as libc::c_int)
                            || __c > 255 as libc::c_int
                        {
                            __c
                        } else {
                            *(*__ctype_toupper_loc()).offset(__c as isize)
                        };
                    } else {
                        __res = toupper(w as libc::c_int);
                    }
                } else {
                    __res = *(*__ctype_toupper_loc()).offset(w as libc::c_int as isize);
                }
                __res
            }) as libc::c_char;
        } else {
            nletter += 1;
            nletter;
        }
        *out.offset(i as isize) = w;
        i += 1;
        i;
    }
    *out.offset(len as isize) = '\0' as i32 as libc::c_char;
    if nletter == len {
        let mut i_0: libc::c_int = 0 as libc::c_int;
        while i_0 < len / 2 as libc::c_int {
            let mut temp: libc::c_char = *out.offset(i_0 as isize);
            *out
                .offset(
                    i_0 as isize,
                ) = *out.offset((len - i_0 - 1 as libc::c_int) as isize);
            *out.offset((len - i_0 - 1 as libc::c_int) as isize) = temp;
            i_0 += 1;
            i_0;
        }
    }
    return out;
}
