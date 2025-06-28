use crate::binn_struct::BinnStruct;

impl BinnStruct {
    pub fn binn_create(&mut self, type_: i32, size: i32) -> bool {
        let alloc_size = match size {
            s if s > 0 => s,
            _ => 256,
        };
        let allocated_memory = unsafe { libc::malloc(alloc_size as usize) };
        if allocated_memory.is_null() {
            return false;
        }
        self.header = 0x21; // 初始化header
        self.allocated = true;
        self.pbuf = Some(allocated_memory);
        self.ptr = allocated_memory;
        self.alloc_size = alloc_size;
        true
    }
}