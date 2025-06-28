use ::libc;
extern "C" {
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
pub type size_t = libc::c_ulong;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct KeyValuePair {
    pub key: *mut libc::c_char,
    pub value: *mut libc::c_char,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct Map {
    pub pairs: *mut KeyValuePair,
    pub size: size_t,
}
#[no_mangle]
pub unsafe extern "C" fn check_dict_case(mut dict: Map) -> bool {
    if dict.size == 0 as libc::c_int as libc::c_ulong {
        return 0 as libc::c_int != 0;
    }
    let mut islower: libc::c_int = 0 as libc::c_int;
    let mut isupper: libc::c_int = 0 as libc::c_int;
    let mut i: size_t = 0 as libc::c_int as size_t;
    while i < dict.size {
        let mut key: *mut libc::c_char = (*(dict.pairs).offset(i as isize)).key;
        let mut j: libc::c_int = 0 as libc::c_int;
        while (j as libc::c_ulong) < strlen(key) {
            if (*key.offset(j as isize) as libc::c_int) < 65 as libc::c_int
                || *key.offset(j as isize) as libc::c_int > 90 as libc::c_int
                    && (*key.offset(j as isize) as libc::c_int) < 97 as libc::c_int
                || *key.offset(j as isize) as libc::c_int > 122 as libc::c_int
            {
                return 0 as libc::c_int != 0;
            }
            if *key.offset(j as isize) as libc::c_int >= 65 as libc::c_int
                && *key.offset(j as isize) as libc::c_int <= 90 as libc::c_int
            {
                isupper = 1 as libc::c_int;
            }
            if *key.offset(j as isize) as libc::c_int >= 97 as libc::c_int
                && *key.offset(j as isize) as libc::c_int <= 122 as libc::c_int
            {
                islower = 1 as libc::c_int;
            }
            if isupper + islower == 2 as libc::c_int {
                return 0 as libc::c_int != 0;
            }
            j += 1;
            j;
        }
        i = i.wrapping_add(1);
        i;
    }
    return 1 as libc::c_int != 0;
}
