use ::libc;
extern "C" {
    pub type _IO_wide_data;
    pub type _IO_codecvt;
    pub type _IO_marker;
    static mut stderr: *mut FILE;
    fn fprintf(_: *mut FILE, _: *const libc::c_char, _: ...) -> libc::c_int;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn exit(_: libc::c_int) -> !;
    fn qsort(
        __base: *mut libc::c_void,
        __nmemb: size_t,
        __size: size_t,
        __compar: __compar_fn_t,
    );
}
pub type size_t = libc::c_ulong;
pub type __off_t = libc::c_long;
pub type __off64_t = libc::c_long;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _IO_FILE {
    pub _flags: libc::c_int,
    pub _IO_read_ptr: *mut libc::c_char,
    pub _IO_read_end: *mut libc::c_char,
    pub _IO_read_base: *mut libc::c_char,
    pub _IO_write_base: *mut libc::c_char,
    pub _IO_write_ptr: *mut libc::c_char,
    pub _IO_write_end: *mut libc::c_char,
    pub _IO_buf_base: *mut libc::c_char,
    pub _IO_buf_end: *mut libc::c_char,
    pub _IO_save_base: *mut libc::c_char,
    pub _IO_backup_base: *mut libc::c_char,
    pub _IO_save_end: *mut libc::c_char,
    pub _markers: *mut _IO_marker,
    pub _chain: *mut _IO_FILE,
    pub _fileno: libc::c_int,
    pub _flags2: libc::c_int,
    pub _old_offset: __off_t,
    pub _cur_column: libc::c_ushort,
    pub _vtable_offset: libc::c_schar,
    pub _shortbuf: [libc::c_char; 1],
    pub _lock: *mut libc::c_void,
    pub _offset: __off64_t,
    pub _codecvt: *mut _IO_codecvt,
    pub _wide_data: *mut _IO_wide_data,
    pub _freeres_list: *mut _IO_FILE,
    pub _freeres_buf: *mut libc::c_void,
    pub __pad5: size_t,
    pub _mode: libc::c_int,
    pub _unused2: [libc::c_char; 20],
}
pub type _IO_lock_t = ();
pub type FILE = _IO_FILE;
pub type __compar_fn_t = Option::<
    unsafe extern "C" fn(*const libc::c_void, *const libc::c_void) -> libc::c_int,
>;
#[no_mangle]
pub unsafe extern "C" fn compare(
    mut a: *const libc::c_void,
    mut b: *const libc::c_void,
) -> libc::c_int {
    return *(a as *mut libc::c_int) - *(b as *mut libc::c_int);
}
#[no_mangle]
pub unsafe extern "C" fn strange_sort_list(
    mut lst: *mut libc::c_int,
    mut size: libc::c_int,
    mut result_size: *mut libc::c_int,
) -> *mut libc::c_int {
    qsort(
        lst as *mut libc::c_void,
        size as size_t,
        ::core::mem::size_of::<libc::c_int>() as libc::c_ulong,
        Some(
            compare
                as unsafe extern "C" fn(
                    *const libc::c_void,
                    *const libc::c_void,
                ) -> libc::c_int,
        ),
    );
    let mut out: *mut libc::c_int = malloc(
        (size as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
    ) as *mut libc::c_int;
    if out.is_null() {
        fprintf(
            stderr,
            b"Memory allocation failed\n\0" as *const u8 as *const libc::c_char,
        );
        exit(1 as libc::c_int);
    }
    let mut l: libc::c_int = 0 as libc::c_int;
    let mut r: libc::c_int = size - 1 as libc::c_int;
    let mut index: libc::c_int = 0 as libc::c_int;
    while l < r {
        let fresh0 = l;
        l = l + 1;
        let fresh1 = index;
        index = index + 1;
        *out.offset(fresh1 as isize) = *lst.offset(fresh0 as isize);
        let fresh2 = r;
        r = r - 1;
        let fresh3 = index;
        index = index + 1;
        *out.offset(fresh3 as isize) = *lst.offset(fresh2 as isize);
    }
    if l == r {
        let fresh4 = index;
        index = index + 1;
        *out.offset(fresh4 as isize) = *lst.offset(l as isize);
    }
    *result_size = size;
    return out;
}
