use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
    fn MD5_Init(c: *mut MD5_CTX) -> libc::c_int;
    fn MD5_Update(
        c: *mut MD5_CTX,
        data: *const libc::c_void,
        len: size_t,
    ) -> libc::c_int;
    fn MD5_Final(md: *mut libc::c_uchar, c: *mut MD5_CTX) -> libc::c_int;
}
pub type size_t = libc::c_ulong;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct MD5state_st {
    pub A: libc::c_uint,
    pub B: libc::c_uint,
    pub C: libc::c_uint,
    pub D: libc::c_uint,
    pub Nl: libc::c_uint,
    pub Nh: libc::c_uint,
    pub data: [libc::c_uint; 16],
    pub num: libc::c_uint,
}
pub type MD5_CTX = MD5state_st;
#[no_mangle]
pub unsafe extern "C" fn string_to_md5(
    mut text: *const libc::c_char,
) -> *mut libc::c_char {
    let mut md: [libc::c_uchar; 16] = [0; 16];
    if strlen(text) == 0 as libc::c_int as libc::c_ulong {
        return b"None\0" as *const u8 as *const libc::c_char as *mut libc::c_char;
    }
    let mut c: MD5_CTX = MD5_CTX {
        A: 0,
        B: 0,
        C: 0,
        D: 0,
        Nl: 0,
        Nh: 0,
        data: [0; 16],
        num: 0,
    };
    MD5_Init(&mut c);
    MD5_Update(&mut c, text as *mut libc::c_uchar as *const libc::c_void, strlen(text));
    MD5_Final(md.as_mut_ptr(), &mut c);
    let mut out_str: *mut libc::c_char = malloc(33 as libc::c_int as libc::c_ulong)
        as *mut libc::c_char;
    if out_str.is_null() {
        return 0 as *mut libc::c_char;
    }
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < 16 as libc::c_int {
        let mut w: libc::c_char = 0;
        if (md[i as usize] as libc::c_int) < 160 as libc::c_int {
            w = (48 as libc::c_int + md[i as usize] as libc::c_int / 16 as libc::c_int)
                as libc::c_char;
        } else {
            w = (87 as libc::c_int + md[i as usize] as libc::c_int / 16 as libc::c_int)
                as libc::c_char;
        }
        *out_str.offset((2 as libc::c_int * i) as isize) = w;
        if (md[i as usize] as libc::c_int % 16 as libc::c_int) < 10 as libc::c_int {
            w = (48 as libc::c_int + md[i as usize] as libc::c_int % 16 as libc::c_int)
                as libc::c_char;
        } else {
            w = (87 as libc::c_int + md[i as usize] as libc::c_int % 16 as libc::c_int)
                as libc::c_char;
        }
        *out_str.offset((2 as libc::c_int * i + 1 as libc::c_int) as isize) = w;
        i += 1;
        i;
    }
    *out_str.offset(32 as libc::c_int as isize) = '\0' as i32 as libc::c_char;
    return out_str;
}
