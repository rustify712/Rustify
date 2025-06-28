// Utilities translated from C to Rust

pub fn copy_be16(dest: &mut u16, source: &u16) {
    *dest = source.to_be();
}

pub fn copy_be32(dest: &mut u32, source: &u32) {
    *dest = source.to_be();
}

pub fn copy_be64(dest: &mut u64, source: &u64) {
    *dest = source.to_be();
}
