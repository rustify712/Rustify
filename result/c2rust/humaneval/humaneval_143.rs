use ::libc;
extern "C" {
    fn strcat(_: *mut libc::c_char, _: *const libc::c_char) -> *mut libc::c_char;
    fn strlen(_: *const libc::c_char) -> libc::c_ulong;
    fn malloc(_: libc::c_ulong) -> *mut libc::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn words_in_sentence(
    mut sentence: *const libc::c_char,
) -> *mut libc::c_char {
    let mut out: *mut libc::c_char = malloc(
        (100 as libc::c_int as libc::c_ulong)
            .wrapping_mul(::core::mem::size_of::<libc::c_char>() as libc::c_ulong),
    ) as *mut libc::c_char;
    *out.offset(0 as libc::c_int as isize) = '\0' as i32 as libc::c_char;
    let mut current: [libc::c_char; 100] = [0; 100];
    let mut out_index: libc::c_int = 0 as libc::c_int;
    let mut sentence_len: libc::c_int = strlen(sentence) as libc::c_int;
    let mut current_index: libc::c_int = 0 as libc::c_int;
    let mut i: libc::c_int = 0 as libc::c_int;
    while i <= sentence_len {
        if *sentence.offset(i as isize) as libc::c_int != ' ' as i32
            && *sentence.offset(i as isize) as libc::c_int != '\0' as i32
        {
            let fresh0 = current_index;
            current_index = current_index + 1;
            current[fresh0 as usize] = *sentence.offset(i as isize);
        } else {
            current[current_index as usize] = '\0' as i32 as libc::c_char;
            let mut isp: bool = 1 as libc::c_int != 0;
            let mut l: libc::c_int = current_index;
            if l < 2 as libc::c_int {
                isp = 0 as libc::c_int != 0;
            }
            let mut j: libc::c_int = 2 as libc::c_int;
            while j * j <= l {
                if l % j == 0 as libc::c_int {
                    isp = 0 as libc::c_int != 0;
                    break;
                } else {
                    j += 1;
                    j;
                }
            }
            if isp {
                strcat(out, current.as_mut_ptr());
                strcat(out, b" \0" as *const u8 as *const libc::c_char);
            }
            current_index = 0 as libc::c_int;
        }
        i += 1;
        i;
    }
    if strlen(out) > 0 as libc::c_int as libc::c_ulong {
        *out
            .offset(
                (strlen(out)).wrapping_sub(1 as libc::c_int as libc::c_ulong) as isize,
            ) = '\0' as i32 as libc::c_char;
    }
    return out;
}
