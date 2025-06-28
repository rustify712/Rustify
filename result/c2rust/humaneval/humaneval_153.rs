use ::libc;
extern "C" {
    fn snprintf(
        _: *mut libc::c_char,
        _: libc::c_ulong,
        _: *const libc::c_char,
        _: ...
    ) -> libc::c_int;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[no_mangle]
pub unsafe extern "C" fn Strongest_Extension(
    mut class_name: *const libc::c_char,
    mut extensions: *mut *const libc::c_char,
    mut extensions_count: libc::c_int,
) -> *mut libc::c_char {
    let mut strongest: *mut libc::c_char = 0 as *mut libc::c_char;
    let mut max_strength: libc::c_int = -(1000 as libc::c_int);
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < extensions_count {
        let mut strength: libc::c_int = 0 as libc::c_int;
        let mut ext: *const libc::c_char = *extensions.offset(i as isize);
        let mut len: libc::c_int = strlen(ext) as libc::c_int;
        let mut j: libc::c_int = 0 as libc::c_int;
        while j < len {
            let mut chr: libc::c_char = *ext.offset(j as isize);
            if chr as libc::c_int >= 'A' as i32 && chr as libc::c_int <= 'Z' as i32 {
                strength += 1 as libc::c_int;
            }
            if chr as libc::c_int >= 'a' as i32 && chr as libc::c_int <= 'z' as i32 {
                strength -= 1 as libc::c_int;
            }
            j += 1;
            j;
        }
        if strength > max_strength {
            max_strength = strength;
            strongest = ext as *mut libc::c_char;
        }
        i += 1;
        i;
    }
    static mut result: [libc::c_char; 256] = [0; 256];
    snprintf(
        result.as_mut_ptr(),
        ::core::mem::size_of::<[libc::c_char; 256]>() as libc::c_ulong,
        b"%s.%s\0" as *const u8 as *const libc::c_char,
        class_name,
        strongest,
    );
    return result.as_mut_ptr();
}
