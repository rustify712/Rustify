// Rust implementation of ArrayList
// Translated from C version

use std::alloc::{alloc, dealloc, Layout};
use std::ptr;

pub struct ArrayList<T> {
    data: *mut T,
    length: usize,
    _alloced: usize,
}

impl<T> ArrayList<T> {
    pub fn new(length: usize) -> Option<Self> {
        let size = if length <= 0 { 16 } else { length };
        
        let layout = Layout::array::<T>(size).unwrap();
        let data = unsafe { alloc(layout) as *mut T };
        
        if data.is_null() {
            None
        } else {
            Some(ArrayList {
                data,
                length: 0,
                _alloced: size,
            })
        }
    }

    fn enlarge(&mut self) -> bool {
        let newsize = self._alloced * 2;
        let new_layout = Layout::array::<T>(newsize).unwrap();
        
        unsafe {
            let new_data = alloc(new_layout) as *mut T;
            if new_data.is_null() {
                return false;
            }
            
            ptr::copy_nonoverlapping(self.data, new_data, self.length);
            
            let old_layout = Layout::array::<T>(self._alloced).unwrap();
            dealloc(self.data as *mut u8, old_layout);
            
            self.data = new_data;
            self._alloced = newsize;
            true
        }
    }
}

impl<T> Drop for ArrayList<T> {
    fn drop(&mut self) {
        if !self.data.is_null() {
            let layout = Layout::array::<T>(self._alloced).unwrap();
            unsafe { dealloc(self.data as *mut u8, layout); }
        }
    }
}