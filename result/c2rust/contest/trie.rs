use ::libc;
extern "C" {
    fn alloc_test_malloc(bytes: size_t) -> *mut libc::c_void;
    fn alloc_test_free(ptr: *mut libc::c_void);
    fn alloc_test_calloc(nmemb: size_t, bytes: size_t) -> *mut libc::c_void;
}
pub type size_t = libc::c_ulong;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _Trie {
    pub root_node: *mut TrieNode,
}
pub type TrieNode = _TrieNode;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _TrieNode {
    pub data: TrieValue,
    pub use_count: libc::c_uint,
    pub next: [*mut TrieNode; 256],
}
pub type TrieValue = *mut libc::c_void;
pub type Trie = _Trie;
#[no_mangle]
pub unsafe extern "C" fn trie_new() -> *mut Trie {
    let mut new_trie: *mut Trie = 0 as *mut Trie;
    new_trie = alloc_test_malloc(::core::mem::size_of::<Trie>() as libc::c_ulong)
        as *mut Trie;
    if new_trie.is_null() {
        return 0 as *mut Trie;
    }
    (*new_trie).root_node = 0 as *mut TrieNode;
    return new_trie;
}
unsafe extern "C" fn trie_free_list_push(
    mut list: *mut *mut TrieNode,
    mut node: *mut TrieNode,
) {
    (*node).data = *list as TrieValue;
    *list = node;
}
unsafe extern "C" fn trie_free_list_pop(mut list: *mut *mut TrieNode) -> *mut TrieNode {
    let mut result: *mut TrieNode = 0 as *mut TrieNode;
    result = *list;
    *list = (*result).data as *mut TrieNode;
    return result;
}
#[no_mangle]
pub unsafe extern "C" fn trie_free(mut trie: *mut Trie) {
    let mut free_list: *mut TrieNode = 0 as *mut TrieNode;
    let mut node: *mut TrieNode = 0 as *mut TrieNode;
    let mut i: libc::c_int = 0;
    free_list = 0 as *mut TrieNode;
    if !((*trie).root_node).is_null() {
        trie_free_list_push(&mut free_list, (*trie).root_node);
    }
    while !free_list.is_null() {
        node = trie_free_list_pop(&mut free_list);
        i = 0 as libc::c_int;
        while i < 256 as libc::c_int {
            if !((*node).next[i as usize]).is_null() {
                trie_free_list_push(&mut free_list, (*node).next[i as usize]);
            }
            i += 1;
            i;
        }
        alloc_test_free(node as *mut libc::c_void);
    }
    alloc_test_free(trie as *mut libc::c_void);
}
unsafe extern "C" fn trie_find_end(
    mut trie: *mut Trie,
    mut key: *mut libc::c_char,
) -> *mut TrieNode {
    let mut node: *mut TrieNode = 0 as *mut TrieNode;
    let mut p: *mut libc::c_char = 0 as *mut libc::c_char;
    node = (*trie).root_node;
    p = key;
    while *p as libc::c_int != '\0' as i32 {
        if node.is_null() {
            return 0 as *mut TrieNode;
        }
        node = (*node).next[*p as libc::c_uchar as usize];
        p = p.offset(1);
        p;
    }
    return node;
}
unsafe extern "C" fn trie_find_end_binary(
    mut trie: *mut Trie,
    mut key: *mut libc::c_uchar,
    mut key_length: libc::c_int,
) -> *mut TrieNode {
    let mut node: *mut TrieNode = 0 as *mut TrieNode;
    let mut j: libc::c_int = 0;
    let mut c: libc::c_int = 0;
    node = (*trie).root_node;
    j = 0 as libc::c_int;
    while j < key_length {
        if node.is_null() {
            return 0 as *mut TrieNode;
        }
        c = *key.offset(j as isize) as libc::c_int;
        node = (*node).next[c as usize];
        j += 1;
        j;
    }
    return node;
}
unsafe extern "C" fn trie_insert_rollback(
    mut trie: *mut Trie,
    mut key: *mut libc::c_uchar,
) {
    let mut node: *mut TrieNode = 0 as *mut TrieNode;
    let mut prev_ptr: *mut *mut TrieNode = 0 as *mut *mut TrieNode;
    let mut next_node: *mut TrieNode = 0 as *mut TrieNode;
    let mut next_prev_ptr: *mut *mut TrieNode = 0 as *mut *mut TrieNode;
    let mut p: *mut libc::c_uchar = 0 as *mut libc::c_uchar;
    node = (*trie).root_node;
    prev_ptr = &mut (*trie).root_node;
    p = key;
    while !node.is_null() {
        next_prev_ptr = &mut *((*node).next).as_mut_ptr().offset(*p as isize)
            as *mut *mut TrieNode;
        next_node = *next_prev_ptr;
        p = p.offset(1);
        p;
        (*node).use_count = ((*node).use_count).wrapping_sub(1);
        (*node).use_count;
        if (*node).use_count == 0 as libc::c_int as libc::c_uint {
            alloc_test_free(node as *mut libc::c_void);
            if !prev_ptr.is_null() {
                *prev_ptr = 0 as *mut TrieNode;
            }
            next_prev_ptr = 0 as *mut *mut TrieNode;
        }
        node = next_node;
        prev_ptr = next_prev_ptr;
    }
}
#[no_mangle]
pub unsafe extern "C" fn trie_insert(
    mut trie: *mut Trie,
    mut key: *mut libc::c_char,
    mut value: TrieValue,
) -> libc::c_int {
    let mut rover: *mut *mut TrieNode = 0 as *mut *mut TrieNode;
    let mut node: *mut TrieNode = 0 as *mut TrieNode;
    let mut p: *mut libc::c_char = 0 as *mut libc::c_char;
    let mut c: libc::c_int = 0;
    if value.is_null() {
        return 0 as libc::c_int;
    }
    node = trie_find_end(trie, key);
    if !node.is_null() && !((*node).data).is_null() {
        (*node).data = value;
        return 1 as libc::c_int;
    }
    rover = &mut (*trie).root_node;
    p = key;
    loop {
        node = *rover;
        if node.is_null() {
            node = alloc_test_calloc(
                1 as libc::c_int as size_t,
                ::core::mem::size_of::<TrieNode>() as libc::c_ulong,
            ) as *mut TrieNode;
            if node.is_null() {
                trie_insert_rollback(trie, key as *mut libc::c_uchar);
                return 0 as libc::c_int;
            }
            (*node).data = 0 as *mut libc::c_void;
            *rover = node;
        }
        (*node).use_count = ((*node).use_count).wrapping_add(1);
        (*node).use_count;
        c = *p as libc::c_uchar as libc::c_int;
        if c == '\0' as i32 {
            (*node).data = value;
            break;
        } else {
            rover = &mut *((*node).next).as_mut_ptr().offset(c as isize)
                as *mut *mut TrieNode;
            p = p.offset(1);
            p;
        }
    }
    return 1 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn trie_insert_binary(
    mut trie: *mut Trie,
    mut key: *mut libc::c_uchar,
    mut key_length: libc::c_int,
    mut value: TrieValue,
) -> libc::c_int {
    let mut rover: *mut *mut TrieNode = 0 as *mut *mut TrieNode;
    let mut node: *mut TrieNode = 0 as *mut TrieNode;
    let mut p: libc::c_int = 0;
    let mut c: libc::c_int = 0;
    if value.is_null() {
        return 0 as libc::c_int;
    }
    node = trie_find_end_binary(trie, key, key_length);
    if !node.is_null() && !((*node).data).is_null() {
        (*node).data = value;
        return 1 as libc::c_int;
    }
    rover = &mut (*trie).root_node;
    p = 0 as libc::c_int;
    loop {
        node = *rover;
        if node.is_null() {
            node = alloc_test_calloc(
                1 as libc::c_int as size_t,
                ::core::mem::size_of::<TrieNode>() as libc::c_ulong,
            ) as *mut TrieNode;
            if node.is_null() {
                trie_insert_rollback(trie, key);
                return 0 as libc::c_int;
            }
            (*node).data = 0 as *mut libc::c_void;
            *rover = node;
        }
        (*node).use_count = ((*node).use_count).wrapping_add(1);
        (*node).use_count;
        c = *key.offset(p as isize) as libc::c_int;
        if p == key_length {
            (*node).data = value;
            break;
        } else {
            rover = &mut *((*node).next).as_mut_ptr().offset(c as isize)
                as *mut *mut TrieNode;
            p += 1;
            p;
        }
    }
    return 1 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn trie_remove_binary(
    mut trie: *mut Trie,
    mut key: *mut libc::c_uchar,
    mut key_length: libc::c_int,
) -> libc::c_int {
    let mut node: *mut TrieNode = 0 as *mut TrieNode;
    let mut next: *mut TrieNode = 0 as *mut TrieNode;
    let mut last_next_ptr: *mut *mut TrieNode = 0 as *mut *mut TrieNode;
    let mut p: libc::c_int = 0;
    let mut c: libc::c_int = 0;
    node = trie_find_end_binary(trie, key, key_length);
    if !node.is_null() && !((*node).data).is_null() {
        (*node).data = 0 as *mut libc::c_void;
    } else {
        return 0 as libc::c_int
    }
    node = (*trie).root_node;
    last_next_ptr = &mut (*trie).root_node;
    p = 0 as libc::c_int;
    loop {
        c = *key.offset(p as isize) as libc::c_int;
        next = (*node).next[c as usize];
        (*node).use_count = ((*node).use_count).wrapping_sub(1);
        (*node).use_count;
        if (*node).use_count <= 0 as libc::c_int as libc::c_uint {
            alloc_test_free(node as *mut libc::c_void);
            if !last_next_ptr.is_null() {
                *last_next_ptr = 0 as *mut TrieNode;
                last_next_ptr = 0 as *mut *mut TrieNode;
            }
        }
        if p == key_length {
            break;
        }
        p += 1;
        p;
        if !last_next_ptr.is_null() {
            last_next_ptr = &mut *((*node).next).as_mut_ptr().offset(c as isize)
                as *mut *mut TrieNode;
        }
        node = next;
    }
    return 1 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn trie_remove(
    mut trie: *mut Trie,
    mut key: *mut libc::c_char,
) -> libc::c_int {
    let mut node: *mut TrieNode = 0 as *mut TrieNode;
    let mut next: *mut TrieNode = 0 as *mut TrieNode;
    let mut last_next_ptr: *mut *mut TrieNode = 0 as *mut *mut TrieNode;
    let mut p: *mut libc::c_char = 0 as *mut libc::c_char;
    let mut c: libc::c_int = 0;
    node = trie_find_end(trie, key);
    if !node.is_null() && !((*node).data).is_null() {
        (*node).data = 0 as *mut libc::c_void;
    } else {
        return 0 as libc::c_int
    }
    node = (*trie).root_node;
    last_next_ptr = &mut (*trie).root_node;
    p = key;
    loop {
        c = *p as libc::c_uchar as libc::c_int;
        next = (*node).next[c as usize];
        (*node).use_count = ((*node).use_count).wrapping_sub(1);
        (*node).use_count;
        if (*node).use_count <= 0 as libc::c_int as libc::c_uint {
            alloc_test_free(node as *mut libc::c_void);
            if !last_next_ptr.is_null() {
                *last_next_ptr = 0 as *mut TrieNode;
                last_next_ptr = 0 as *mut *mut TrieNode;
            }
        }
        if c == '\0' as i32 {
            break;
        }
        p = p.offset(1);
        p;
        if !last_next_ptr.is_null() {
            last_next_ptr = &mut *((*node).next).as_mut_ptr().offset(c as isize)
                as *mut *mut TrieNode;
        }
        node = next;
    }
    return 1 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn trie_lookup(
    mut trie: *mut Trie,
    mut key: *mut libc::c_char,
) -> TrieValue {
    let mut node: *mut TrieNode = 0 as *mut TrieNode;
    node = trie_find_end(trie, key);
    if !node.is_null() { return (*node).data } else { return 0 as *mut libc::c_void };
}
#[no_mangle]
pub unsafe extern "C" fn trie_lookup_binary(
    mut trie: *mut Trie,
    mut key: *mut libc::c_uchar,
    mut key_length: libc::c_int,
) -> TrieValue {
    let mut node: *mut TrieNode = 0 as *mut TrieNode;
    node = trie_find_end_binary(trie, key, key_length);
    if !node.is_null() { return (*node).data } else { return 0 as *mut libc::c_void };
}
#[no_mangle]
pub unsafe extern "C" fn trie_num_entries(mut trie: *mut Trie) -> libc::c_uint {
    if ((*trie).root_node).is_null() {
        return 0 as libc::c_int as libc::c_uint
    } else {
        return (*(*trie).root_node).use_count
    };
}
