use ::libc;
extern "C" {
    fn realloc(_: *mut libc::c_void, _: libc::c_ulong) -> *mut libc::c_void;
    fn strncmp(
        _: *const libc::c_char,
        _: *const libc::c_char,
        _: libc::c_ulong,
    ) -> libc::c_int;
    fn strdup(_: *const libc::c_char) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct StringArray {
    pub data: *mut *mut libc::c_char,
    pub size: libc::c_int,
}
#[no_mangle]
pub unsafe extern "C" fn filter_by_prefix(
    mut strings: StringArray,
    mut prefix: *const libc::c_char,
) -> StringArray {
    let mut out: StringArray = {
        let mut init = StringArray {
            data: 0 as *mut *mut libc::c_char,
            size: 0 as libc::c_int,
        };
        init
    };
    let mut prefix_len: libc::c_int = strlen(prefix) as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < strings.size {
        if strncmp(
            *(strings.data).offset(i as isize),
            prefix,
            prefix_len as libc::c_ulong,
        ) == 0 as libc::c_int
        {
            out.size += 1;
            out.size;
            out
                .data = realloc(
                out.data as *mut libc::c_void,
                (out.size as libc::c_ulong)
                    .wrapping_mul(
                        ::core::mem::size_of::<*mut libc::c_char>() as libc::c_ulong,
                    ),
            ) as *mut *mut libc::c_char;
            let ref mut fresh0 = *(out.data)
                .offset((out.size - 1 as libc::c_int) as isize);
            *fresh0 = strdup(*(strings.data).offset(i as isize));
        }
        i += 1;
        i;
    }
    return out;
}
