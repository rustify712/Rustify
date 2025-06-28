use ::libc;
#[no_mangle]
pub unsafe extern "C" fn car_race_collision(mut n: libc::c_int) -> libc::c_int {
    return n * n;
}
