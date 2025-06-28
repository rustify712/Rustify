// Rust实现双向链表
// 从C版本转换而来

use std::ptr;

pub type ListValue = i32;

pub struct ListEntry {
    pub data: ListValue,
    prev: *mut ListEntry,
    next: *mut ListEntry,
}

pub struct List {
    head: *mut ListEntry,
    tail: *mut ListEntry,
    length: usize,
}

impl List {
    pub fn new() -> Self {
        List {
            head: ptr::null_mut(),
            tail: ptr::null_mut(),
            length: 0,
        }
    }

    pub fn prepend(&mut self, data: ListValue) -> *mut ListEntry {
        let new_entry = Box::into_raw(Box::new(ListEntry {
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
            self.length += 1;
        }

        new_entry
    }

    pub fn append(&mut self, data: ListValue) -> *mut ListEntry {
        let new_entry = Box::into_raw(Box::new(ListEntry {
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
            self.length += 1;
        }

        new_entry
    }

    pub fn remove(&mut self, entry: *mut ListEntry) {
        if entry.is_null() {
            return;
        }

        unsafe {
            // 更新前驱节点的next指针
            if !(*entry).prev.is_null() {
                (*(*entry).prev).next = (*entry).next;
            } else {
                self.head = (*entry).next;
            }

            // 更新后继节点的prev指针
            if !(*entry).next.is_null() {
                (*(*entry).next).prev = (*entry).prev;
            } else {
                self.tail = (*entry).prev;
            }

            // 释放节点内存
            Box::from_raw(entry);
            self.length -= 1;
        }
    }
}