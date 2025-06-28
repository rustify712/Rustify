use ::libc;
extern "C" {
    fn realloc(_: *mut libc::c_void, _: libc::c_ulong) -> *mut libc::c_void;
}
pub type size_t = libc::c_ulong;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct any {
    pub type_0: C2RustUnnamed_0,
    pub c2rust_unnamed: C2RustUnnamed,
}
#[derive(Copy, Clone)]
#[repr(C)]
pub union C2RustUnnamed {
    pub int_value: libc::c_int,
    pub double_value: libc::c_double,
    pub string_value: *mut libc::c_char,
    pub other_value: *mut libc::c_void,
}
pub type C2RustUnnamed_0 = libc::c_uint;
pub const OTHER: C2RustUnnamed_0 = 3;
pub const STRING: C2RustUnnamed_0 = 2;
pub const DOUBLE: C2RustUnnamed_0 = 1;
pub const INT: C2RustUnnamed_0 = 0;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct list_any {
    pub items: *mut any,
    pub size: size_t,
    pub capacity: size_t,
}
#[no_mangle]
pub unsafe extern "C" fn filter_integers(
    mut values: list_any,
    mut out_size: *mut size_t,
) -> *mut libc::c_int {
    let mut out: *mut libc::c_int = 0 as *mut libc::c_int;
    let mut count: size_t = 0 as libc::c_int as size_t;
    let mut i: size_t = 0 as libc::c_int as size_t;
    while i < values.size {
        if (*(values.items).offset(i as isize)).type_0 as libc::c_uint
            == INT as libc::c_int as libc::c_uint
        {
            out = realloc(
                out as *mut libc::c_void,
                count
                    .wrapping_add(1 as libc::c_int as libc::c_ulong)
                    .wrapping_mul(::core::mem::size_of::<libc::c_int>() as libc::c_ulong),
            ) as *mut libc::c_int;
            *out
                .offset(
                    count as isize,
                ) = (*(values.items).offset(i as isize)).c2rust_unnamed.int_value;
            count = count.wrapping_add(1);
            count;
        }
        i = i.wrapping_add(1);
        i;
    }
    *out_size = count;
    return out;
}
