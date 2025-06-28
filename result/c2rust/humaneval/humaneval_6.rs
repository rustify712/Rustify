use ::libc;
extern "C" {
    fn realloc(_: *mut libc::c_void, _: libc::c_ulong) -> *mut libc::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn parse_nested_parens(
    mut paren_string: *const libc::c_char,
    mut result_size: *mut libc::c_int,
) -> *mut libc::c_int {
    let mut all_levels: *mut libc::c_int = 0 as *mut libc::c_int;
    let mut capacity: libc::c_int = 0 as libc::c_int;
    let mut size: libc::c_int = 0 as libc::c_int;
    let mut level: libc::c_int = 0 as libc::c_int;
    let mut max_level: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0;
    i = 0 as libc::c_int;
    while *paren_string.offset(i as isize) as libc::c_int != '\0' as i32 {
        let mut chr: libc::c_char = *paren_string.offset(i as isize);
        if chr as libc::c_int == '(' as i32 {
            level += 1 as libc::c_int;
            if level > max_level {
                max_level = level;
            }
        } else if chr as libc::c_int == ')' as i32 {
            level -= 1 as libc::c_int;
            if level == 0 as libc::c_int {
                if size >= capacity {
                    capacity = if capacity == 0 as libc::c_int {
                        1 as libc::c_int
                    } else {
                        capacity * 2 as libc::c_int
                    };
                    all_levels = realloc(
                        all_levels as *mut libc::c_void,
                        (capacity as libc::c_ulong)
                            .wrapping_mul(
                                ::core::mem::size_of::<libc::c_int>() as libc::c_ulong,
                            ),
                    ) as *mut libc::c_int;
                }
                let fresh0 = size;
                size = size + 1;
                *all_levels.offset(fresh0 as isize) = max_level;
                max_level = 0 as libc::c_int;
            }
        }
        i += 1;
        i;
    }
    *result_size = size;
    return all_levels;
}
