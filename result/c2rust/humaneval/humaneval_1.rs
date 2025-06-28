use ::libc;
extern "C" {
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
    fn realloc(_: *mut libc::c_void, _: libc::c_ulong) -> *mut libc::c_void;
    fn free(_: *mut libc::c_void);
    fn strcpy(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
}
#[derive(Copy, Clone)]
#[repr(C)]
pub struct StringArray {
    pub data: *mut *mut libc::c_char,
    pub size: libc::c_int,
    pub capacity: libc::c_int,
}
#[no_mangle]
pub unsafe extern "C" fn init_string_array(
    mut arr: *mut StringArray,
    mut capacity: libc::c_int,
) {
    (*arr)
        .data = malloc(
        (capacity as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<*mut libc::c_char>() as libc::c_ulong),
    ) as *mut *mut libc::c_char;
    (*arr).size = 0 as libc::c_int;
    (*arr).capacity = capacity;
}
#[no_mangle]
pub unsafe extern "C" fn push_back(
    mut arr: *mut StringArray,
    mut str: *const libc::c_char,
) {
    if (*arr).size >= (*arr).capacity {
        (*arr).capacity *= 2 as libc::c_int;
        (*arr)
            .data = realloc(
            (*arr).data as *mut libc::c_void,
            ((*arr).capacity as libc::c_ulong)
                .wrapping_mul(
                    ::core::mem::size_of::<*mut libc::c_char>() as libc::c_ulong,
                ),
        ) as *mut *mut libc::c_char;
    }
    let ref mut fresh0 = *((*arr).data).offset((*arr).size as isize);
    *fresh0 = malloc(
        (strlen(str))
            .wrapping_add(1 as libc::c_int as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_char>() as libc::c_ulong),
    ) as *mut libc::c_char;
    strcpy(*((*arr).data).offset((*arr).size as isize), str);
    (*arr).size += 1;
    (*arr).size;
}
#[no_mangle]
pub unsafe extern "C" fn free_string_array(mut arr: *mut StringArray) {
    let mut i: libc::c_int = 0 as libc::c_int;
    while i < (*arr).size {
        free(*((*arr).data).offset(i as isize) as *mut libc::c_void);
        i += 1;
        i;
    }
    free((*arr).data as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn separate_paren_groups(
    mut paren_string: *const libc::c_char,
) -> StringArray {
    let mut all_parens: StringArray = StringArray {
        data: 0 as *mut *mut libc::c_char,
        size: 0,
        capacity: 0,
    };
    init_string_array(&mut all_parens, 10 as libc::c_int);
    let mut current_paren: *mut libc::c_char = malloc(
        (100 as libc::c_int as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_char>() as libc::c_ulong),
    ) as *mut libc::c_char;
    let mut level: libc::c_int = 0 as libc::c_int;
    let mut current_paren_index: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while *paren_string.offset(i as isize) as libc::c_int != '\0' as i32 {
        let mut chr: libc::c_char = *paren_string.offset(i as isize);
        if !(chr as libc::c_int == ' ' as i32) {
            if chr as libc::c_int == '(' as i32 {
                level += 1;
                level;
                let fresh1 = current_paren_index;
                current_paren_index = current_paren_index + 1;
                *current_paren.offset(fresh1 as isize) = chr;
            }
            if chr as libc::c_int == ')' as i32 {
                level -= 1;
                level;
                let fresh2 = current_paren_index;
                current_paren_index = current_paren_index + 1;
                *current_paren.offset(fresh2 as isize) = chr;
                if level == 0 as libc::c_int {
                    *current_paren
                        .offset(
                            current_paren_index as isize,
                        ) = '\0' as i32 as libc::c_char;
                    push_back(&mut all_parens, current_paren);
                    current_paren_index = 0 as libc::c_int;
                }
            }
        }
        i += 1;
        i;
    }
    free(current_paren as *mut libc::c_void);
    return all_parens;
}
