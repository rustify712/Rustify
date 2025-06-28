use ::libc;
extern "C" {
    fn alloc_test_malloc(bytes: size_t) -> *mut libc::c_void;
    fn alloc_test_free(ptr: *mut libc::c_void);
}
pub type size_t = libc::c_ulong;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _Queue {
    pub head: *mut QueueEntry,
    pub tail: *mut QueueEntry,
}
pub type QueueEntry = _QueueEntry;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _QueueEntry {
    pub data: QueueValue,
    pub prev: *mut QueueEntry,
    pub next: *mut QueueEntry,
}
pub type QueueValue = *mut libc::c_void;
pub type Queue = _Queue;
#[no_mangle]
pub unsafe extern "C" fn queue_new() -> *mut Queue {
    let mut queue: *mut Queue = 0 as *mut Queue;
    queue = alloc_test_malloc(::core::mem::size_of::<Queue>() as libc::c_ulong)
        as *mut Queue;
    if queue.is_null() {
        return 0 as *mut Queue;
    }
    (*queue).head = 0 as *mut QueueEntry;
    (*queue).tail = 0 as *mut QueueEntry;
    return queue;
}
#[no_mangle]
pub unsafe extern "C" fn queue_free(mut queue: *mut Queue) {
    while queue_is_empty(queue) == 0 {
        queue_pop_head(queue);
    }
    alloc_test_free(queue as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn queue_push_head(
    mut queue: *mut Queue,
    mut data: QueueValue,
) -> libc::c_int {
    let mut new_entry: *mut QueueEntry = 0 as *mut QueueEntry;
    new_entry = alloc_test_malloc(::core::mem::size_of::<QueueEntry>() as libc::c_ulong)
        as *mut QueueEntry;
    if new_entry.is_null() {
        return 0 as libc::c_int;
    }
    (*new_entry).data = data;
    (*new_entry).prev = 0 as *mut QueueEntry;
    (*new_entry).next = (*queue).head;
    if ((*queue).head).is_null() {
        (*queue).head = new_entry;
        (*queue).tail = new_entry;
    } else {
        (*(*queue).head).prev = new_entry;
        (*queue).head = new_entry;
    }
    return 1 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn queue_pop_head(mut queue: *mut Queue) -> QueueValue {
    let mut entry: *mut QueueEntry = 0 as *mut QueueEntry;
    let mut result: QueueValue = 0 as *mut libc::c_void;
    if queue_is_empty(queue) != 0 {
        return 0 as *mut libc::c_void;
    }
    entry = (*queue).head;
    (*queue).head = (*entry).next;
    result = (*entry).data;
    if ((*queue).head).is_null() {
        (*queue).tail = 0 as *mut QueueEntry;
    } else {
        (*(*queue).head).prev = 0 as *mut QueueEntry;
    }
    alloc_test_free(entry as *mut libc::c_void);
    return result;
}
#[no_mangle]
pub unsafe extern "C" fn queue_peek_head(mut queue: *mut Queue) -> QueueValue {
    if queue_is_empty(queue) != 0 {
        return 0 as *mut libc::c_void
    } else {
        return (*(*queue).head).data
    };
}
#[no_mangle]
pub unsafe extern "C" fn queue_push_tail(
    mut queue: *mut Queue,
    mut data: QueueValue,
) -> libc::c_int {
    let mut new_entry: *mut QueueEntry = 0 as *mut QueueEntry;
    new_entry = alloc_test_malloc(::core::mem::size_of::<QueueEntry>() as libc::c_ulong)
        as *mut QueueEntry;
    if new_entry.is_null() {
        return 0 as libc::c_int;
    }
    (*new_entry).data = data;
    (*new_entry).prev = (*queue).tail;
    (*new_entry).next = 0 as *mut QueueEntry;
    if ((*queue).tail).is_null() {
        (*queue).head = new_entry;
        (*queue).tail = new_entry;
    } else {
        (*(*queue).tail).next = new_entry;
        (*queue).tail = new_entry;
    }
    return 1 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn queue_pop_tail(mut queue: *mut Queue) -> QueueValue {
    let mut entry: *mut QueueEntry = 0 as *mut QueueEntry;
    let mut result: QueueValue = 0 as *mut libc::c_void;
    if queue_is_empty(queue) != 0 {
        return 0 as *mut libc::c_void;
    }
    entry = (*queue).tail;
    (*queue).tail = (*entry).prev;
    result = (*entry).data;
    if ((*queue).tail).is_null() {
        (*queue).head = 0 as *mut QueueEntry;
    } else {
        (*(*queue).tail).next = 0 as *mut QueueEntry;
    }
    alloc_test_free(entry as *mut libc::c_void);
    return result;
}
#[no_mangle]
pub unsafe extern "C" fn queue_peek_tail(mut queue: *mut Queue) -> QueueValue {
    if queue_is_empty(queue) != 0 {
        return 0 as *mut libc::c_void
    } else {
        return (*(*queue).tail).data
    };
}
#[no_mangle]
pub unsafe extern "C" fn queue_is_empty(mut queue: *mut Queue) -> libc::c_int {
    return ((*queue).head == 0 as *mut libc::c_void as *mut QueueEntry) as libc::c_int;
}
