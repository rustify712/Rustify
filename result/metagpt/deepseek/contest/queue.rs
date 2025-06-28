// Rust实现双端队列
// 从C版本转换而来

use std::ptr;

pub type QueueValue = i32;

struct QueueEntry {
    data: QueueValue,
    prev: *mut QueueEntry,
    next: *mut QueueEntry,
}

pub struct Queue {
    head: *mut QueueEntry,
    tail: *mut QueueEntry,
}

impl Queue {
    pub fn new() -> Self {
        Queue {
            head: ptr::null_mut(),
            tail: ptr::null_mut(),
        }
    }

    pub fn push_head(&mut self, data: QueueValue) -> bool {
        let new_entry = Box::into_raw(Box::new(QueueEntry {
            data,
            prev: ptr::null_mut(),
            next: self.head,
        }));

        unsafe {
            if !self.head.is_null() {
                (*self.head).prev = new_entry;
            } else {
                self.tail = new_entry;
            }
            self.head = new_entry;
        }

        true
    }

    pub fn push_tail(&mut self, data: QueueValue) -> bool {
        let new_entry = Box::into_raw(Box::new(QueueEntry {
            data,
            prev: self.tail,
            next: ptr::null_mut(),
        }));

        unsafe {
            if !self.tail.is_null() {
                (*self.tail).next = new_entry;
            } else {
                self.head = new_entry;
            }
            self.tail = new_entry;
        }

        true
    }

    pub fn pop_head(&mut self) -> Option<QueueValue> {
        if self.head.is_null() {
            return None;
        }

        unsafe {
            let entry = self.head;
            let data = (*entry).data;
            
            self.head = (*entry).next;
            if !self.head.is_null() {
                (*self.head).prev = ptr::null_mut();
            } else {
                self.tail = ptr::null_mut();
            }
            
            Box::from_raw(entry);
            Some(data)
        }
    }

    pub fn pop_tail(&mut self) -> Option<QueueValue> {
        if self.tail.is_null() {
            return None;
        }

        unsafe {
            let entry = self.tail;
            let data = (*entry).data;
            
            self.tail = (*entry).prev;
            if !self.tail.is_null() {
                (*self.tail).next = ptr::null_mut();
            } else {
                self.head = ptr::null_mut();
            }
            
            Box::from_raw(entry);
            Some(data)
        }
    }

    pub fn is_empty(&self) -> bool {
        self.head.is_null()
    }

    pub fn free(&mut self) {
        while let Some(_) = self.pop_head() {}
    }
}