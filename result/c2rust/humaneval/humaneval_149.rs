use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn qsort(
        __base: *mut libc::c_void,
        __nmemb: size_t,
        __size: size_t,
        __compar: __compar_fn_t,
    );
    fn strcmp(_: *const libc::c_char, _: *const libc::c_char) -> libc::c_int;
    fn strdup(_: *const libc::c_char) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
pub type size_t = libc::c_ulong;
pub type __compar_fn_t = Option::<
    unsafe extern "C" fn(*const libc::c_void, *const libc::c_void) -> libc::c_int,
>;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct StringArray {
    pub data: *mut *mut libc::c_char,
    pub size: libc::c_int,
}
#[no_mangle]
pub unsafe extern "C" fn compare2(
    mut a: *const libc::c_void,
    mut b: *const libc::c_void,
) -> libc::c_int {
    let mut str1: *const libc::c_char = *(a as *mut *const libc::c_char);
    let mut str2: *const libc::c_char = *(b as *mut *const libc::c_char);
    let mut len1: libc::c_int = strlen(str1) as libc::c_int;
    let mut len2: libc::c_int = strlen(str2) as libc::c_int;
    if len1 != len2 { return len1 - len2 } else { return strcmp(str1, str2) };
}
#[no_mangle]
pub unsafe extern "C" fn sorted_list_sum(mut lst: StringArray) -> StringArray {
    let mut out: StringArray = StringArray {
        data: 0 as *mut *mut libc::c_char,
        size: 0,
    };
    out
        .data = malloc(
        (lst.size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<*mut libc::c_char>() as libc::c_ulong),
    ) as *mut *mut libc::c_char;
    out.size = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < lst.size {
        if (strlen(*(lst.data).offset(i as isize)))
            .wrapping_rem(2 as libc::c_int as libc::c_ulong)
            == 0 as libc::c_int as libc::c_ulong
        {
            let ref mut fresh0 = *(out.data).offset(out.size as isize);
            *fresh0 = strdup(*(lst.data).offset(i as isize));
            out.size += 1;
            out.size;
        }
        i += 1;
        i;
    }
    qsort(
        out.data as *mut libc::c_void,
        out.size as size_t,
        ::core::mem::size_of::<*mut libc::c_char>() as libc::c_ulong,
        Some(
            compare2
                as unsafe extern "C" fn(
                    *const libc::c_void,
                    *const libc::c_void,
                ) -> libc::c_int,
        ),
    );
    return out;
}
