use ::libc;
extern "C" {
    fn strchr(_: *const libc::c_char, _: libc::c_int) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
    fn __ctype_b_loc() -> *mut *const libc::c_ushort;
    fn __ctype_tolower_loc() -> *mut *const __int32_t;
    fn __ctype_toupper_loc() -> *mut *const __int32_t;
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
pub unsafe extern "C" fn encode(mut message: *const libc::c_char) -> *mut libc::c_char {
    static mut out: [libc::c_char; 256] = [0; 256];
    let mut vowels: *const libc::c_char = b"aeiouAEIOU\0" as *const u8
        as *const libc::c_char;
    let mut len: libc::c_int = strlen(message) as libc::c_int;
    let mut out_index: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < len {
        let mut w: libc::c_char = *message.offset(i as isize);
        if *(*__ctype_b_loc()).offset(w as libc::c_int as isize) as libc::c_int
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
        } else if *(*__ctype_b_loc()).offset(w as libc::c_int as isize) as libc::c_int
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
        }
        if !(strchr(vowels, w as libc::c_int)).is_null() {
            w = (w as libc::c_int + 2 as libc::c_int) as libc::c_char;
        }
        let fresh0 = out_index;
        out_index = out_index + 1;
        out[fresh0 as usize] = w;
        i += 1;
        i;
    }
    out[out_index as usize] = '\0' as i32 as libc::c_char;
    return out.as_mut_ptr();
}
