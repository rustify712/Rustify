use ::libc;
extern "C" {
    fn alloc_test_malloc(bytes: size_t) -> *mut libc::c_void;
    fn alloc_test_free(ptr: *mut libc::c_void);
    fn alloc_test_calloc(nmemb: size_t, bytes: size_t) -> *mut libc::c_void;
}
pub type size_t = libc::c_ulong;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _HashTable {
    pub table: *mut *mut HashTableEntry,
    pub table_size: libc::c_uint,
    pub hash_func: HashTableHashFunc,
    pub equal_func: HashTableEqualFunc,
    pub key_free_func: HashTableKeyFreeFunc,
    pub value_free_func: HashTableValueFreeFunc,
    pub entries: libc::c_uint,
    pub prime_index: libc::c_uint,
}
pub type HashTableValueFreeFunc = Option::<unsafe extern "C" fn(HashTableValue) -> ()>;
pub type HashTableValue = *mut libc::c_void;
pub type HashTableKeyFreeFunc = Option::<unsafe extern "C" fn(HashTableKey) -> ()>;
pub type HashTableKey = *mut libc::c_void;
pub type HashTableEqualFunc = Option::<
    unsafe extern "C" fn(HashTableKey, HashTableKey) -> libc::c_int,
>;
pub type HashTableHashFunc = Option::<
    unsafe extern "C" fn(HashTableKey) -> libc::c_uint,
>;
pub type HashTableEntry = _HashTableEntry;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _HashTableEntry {
    pub pair: HashTablePair,
    pub next: *mut HashTableEntry,
}
pub type HashTablePair = _HashTablePair;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _HashTablePair {
    pub key: HashTableKey,
    pub value: HashTableValue,
}
pub type HashTable = _HashTable;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _HashTableIterator {
    pub hash_table: *mut HashTable,
    pub next_entry: *mut HashTableEntry,
    pub next_chain: libc::c_uint,
}
pub type HashTableIterator = _HashTableIterator;
static mut hash_table_primes: [libc::c_uint; 24] = [
    193 as libc::c_int as libc::c_uint,
    389 as libc::c_int as libc::c_uint,
    769 as libc::c_int as libc::c_uint,
    1543 as libc::c_int as libc::c_uint,
    3079 as libc::c_int as libc::c_uint,
    6151 as libc::c_int as libc::c_uint,
    12289 as libc::c_int as libc::c_uint,
    24593 as libc::c_int as libc::c_uint,
    49157 as libc::c_int as libc::c_uint,
    98317 as libc::c_int as libc::c_uint,
    196613 as libc::c_int as libc::c_uint,
    393241 as libc::c_int as libc::c_uint,
    786433 as libc::c_int as libc::c_uint,
    1572869 as libc::c_int as libc::c_uint,
    3145739 as libc::c_int as libc::c_uint,
    6291469 as libc::c_int as libc::c_uint,
    12582917 as libc::c_int as libc::c_uint,
    25165843 as libc::c_int as libc::c_uint,
    50331653 as libc::c_int as libc::c_uint,
    100663319 as libc::c_int as libc::c_uint,
    201326611 as libc::c_int as libc::c_uint,
    402653189 as libc::c_int as libc::c_uint,
    805306457 as libc::c_int as libc::c_uint,
    1610612741 as libc::c_int as libc::c_uint,
];
static mut hash_table_num_primes: libc::c_uint = 0;
unsafe extern "C" fn hash_table_allocate_table(
    mut hash_table: *mut HashTable,
) -> libc::c_int {
    let mut new_table_size: libc::c_uint = 0;
    if (*hash_table).prime_index < hash_table_num_primes {
        new_table_size = hash_table_primes[(*hash_table).prime_index as usize];
    } else {
        new_table_size = ((*hash_table).entries)
            .wrapping_mul(10 as libc::c_int as libc::c_uint);
    }
    (*hash_table).table_size = new_table_size;
    (*hash_table)
        .table = alloc_test_calloc(
        (*hash_table).table_size as size_t,
        ::core::mem::size_of::<*mut HashTableEntry>() as libc::c_ulong,
    ) as *mut *mut HashTableEntry;
    return ((*hash_table).table != 0 as *mut libc::c_void as *mut *mut HashTableEntry)
        as libc::c_int;
}
unsafe extern "C" fn hash_table_free_entry(
    mut hash_table: *mut HashTable,
    mut entry: *mut HashTableEntry,
) {
    let mut pair: *mut HashTablePair = 0 as *mut HashTablePair;
    pair = &mut (*entry).pair;
    if ((*hash_table).key_free_func).is_some() {
        ((*hash_table).key_free_func).expect("non-null function pointer")((*pair).key);
    }
    if ((*hash_table).value_free_func).is_some() {
        ((*hash_table).value_free_func)
            .expect("non-null function pointer")((*pair).value);
    }
    alloc_test_free(entry as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn hash_table_new(
    mut hash_func: HashTableHashFunc,
    mut equal_func: HashTableEqualFunc,
) -> *mut HashTable {
    let mut hash_table: *mut HashTable = 0 as *mut HashTable;
    hash_table = alloc_test_malloc(::core::mem::size_of::<HashTable>() as libc::c_ulong)
        as *mut HashTable;
    if hash_table.is_null() {
        return 0 as *mut HashTable;
    }
    (*hash_table).hash_func = hash_func;
    (*hash_table).equal_func = equal_func;
    (*hash_table).key_free_func = None;
    (*hash_table).value_free_func = None;
    (*hash_table).entries = 0 as libc::c_int as libc::c_uint;
    (*hash_table).prime_index = 0 as libc::c_int as libc::c_uint;
    if hash_table_allocate_table(hash_table) == 0 {
        alloc_test_free(hash_table as *mut libc::c_void);
        return 0 as *mut HashTable;
    }
    return hash_table;
}
#[no_mangle]
pub unsafe extern "C" fn hash_table_free(mut hash_table: *mut HashTable) {
    let mut rover: *mut HashTableEntry = 0 as *mut HashTableEntry;
    let mut next: *mut HashTableEntry = 0 as *mut HashTableEntry;
    let mut i: libc::c_uint = 0;
    i = 0 as libc::c_int as libc::c_uint;
    while i < (*hash_table).table_size {
        rover = *((*hash_table).table).offset(i as isize);
        while !rover.is_null() {
            next = (*rover).next;
            hash_table_free_entry(hash_table, rover);
            rover = next;
        }
        i = i.wrapping_add(1);
        i;
    }
    alloc_test_free((*hash_table).table as *mut libc::c_void);
    alloc_test_free(hash_table as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn hash_table_register_free_functions(
    mut hash_table: *mut HashTable,
    mut key_free_func: HashTableKeyFreeFunc,
    mut value_free_func: HashTableValueFreeFunc,
) {
    (*hash_table).key_free_func = key_free_func;
    (*hash_table).value_free_func = value_free_func;
}
unsafe extern "C" fn hash_table_enlarge(mut hash_table: *mut HashTable) -> libc::c_int {
    let mut old_table: *mut *mut HashTableEntry = 0 as *mut *mut HashTableEntry;
    let mut old_table_size: libc::c_uint = 0;
    let mut old_prime_index: libc::c_uint = 0;
    let mut rover: *mut HashTableEntry = 0 as *mut HashTableEntry;
    let mut pair: *mut HashTablePair = 0 as *mut HashTablePair;
    let mut next: *mut HashTableEntry = 0 as *mut HashTableEntry;
    let mut index: libc::c_uint = 0;
    let mut i: libc::c_uint = 0;
    old_table = (*hash_table).table;
    old_table_size = (*hash_table).table_size;
    old_prime_index = (*hash_table).prime_index;
    (*hash_table).prime_index = ((*hash_table).prime_index).wrapping_add(1);
    (*hash_table).prime_index;
    if hash_table_allocate_table(hash_table) == 0 {
        (*hash_table).table = old_table;
        (*hash_table).table_size = old_table_size;
        (*hash_table).prime_index = old_prime_index;
        return 0 as libc::c_int;
    }
    i = 0 as libc::c_int as libc::c_uint;
    while i < old_table_size {
        rover = *old_table.offset(i as isize);
        while !rover.is_null() {
            next = (*rover).next;
            pair = &mut (*rover).pair;
            index = (((*hash_table).hash_func)
                .expect("non-null function pointer")((*pair).key))
                .wrapping_rem((*hash_table).table_size);
            (*rover).next = *((*hash_table).table).offset(index as isize);
            let ref mut fresh0 = *((*hash_table).table).offset(index as isize);
            *fresh0 = rover;
            rover = next;
        }
        i = i.wrapping_add(1);
        i;
    }
    alloc_test_free(old_table as *mut libc::c_void);
    return 1 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn hash_table_insert(
    mut hash_table: *mut HashTable,
    mut key: HashTableKey,
    mut value: HashTableValue,
) -> libc::c_int {
    let mut rover: *mut HashTableEntry = 0 as *mut HashTableEntry;
    let mut pair: *mut HashTablePair = 0 as *mut HashTablePair;
    let mut newentry: *mut HashTableEntry = 0 as *mut HashTableEntry;
    let mut index: libc::c_uint = 0;
    if ((*hash_table).entries)
        .wrapping_mul(3 as libc::c_int as libc::c_uint)
        .wrapping_div((*hash_table).table_size) > 0 as libc::c_int as libc::c_uint
    {
        if hash_table_enlarge(hash_table) == 0 {
            return 0 as libc::c_int;
        }
    }
    index = (((*hash_table).hash_func).expect("non-null function pointer")(key))
        .wrapping_rem((*hash_table).table_size);
    rover = *((*hash_table).table).offset(index as isize);
    while !rover.is_null() {
        pair = &mut (*rover).pair;
        if ((*hash_table).equal_func)
            .expect("non-null function pointer")((*pair).key, key) != 0 as libc::c_int
        {
            if ((*hash_table).value_free_func).is_some() {
                ((*hash_table).value_free_func)
                    .expect("non-null function pointer")((*pair).value);
            }
            if ((*hash_table).key_free_func).is_some() {
                ((*hash_table).key_free_func)
                    .expect("non-null function pointer")((*pair).key);
            }
            (*pair).key = key;
            (*pair).value = value;
            return 1 as libc::c_int;
        }
        rover = (*rover).next;
    }
    newentry = alloc_test_malloc(
        ::core::mem::size_of::<HashTableEntry>() as libc::c_ulong,
    ) as *mut HashTableEntry;
    if newentry.is_null() {
        return 0 as libc::c_int;
    }
    (*newentry).pair.key = key;
    (*newentry).pair.value = value;
    (*newentry).next = *((*hash_table).table).offset(index as isize);
    let ref mut fresh1 = *((*hash_table).table).offset(index as isize);
    *fresh1 = newentry;
    (*hash_table).entries = ((*hash_table).entries).wrapping_add(1);
    (*hash_table).entries;
    return 1 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn hash_table_lookup(
    mut hash_table: *mut HashTable,
    mut key: HashTableKey,
) -> HashTableValue {
    let mut rover: *mut HashTableEntry = 0 as *mut HashTableEntry;
    let mut pair: *mut HashTablePair = 0 as *mut HashTablePair;
    let mut index: libc::c_uint = 0;
    index = (((*hash_table).hash_func).expect("non-null function pointer")(key))
        .wrapping_rem((*hash_table).table_size);
    rover = *((*hash_table).table).offset(index as isize);
    while !rover.is_null() {
        pair = &mut (*rover).pair;
        if ((*hash_table).equal_func)
            .expect("non-null function pointer")(key, (*pair).key) != 0 as libc::c_int
        {
            return (*pair).value;
        }
        rover = (*rover).next;
    }
    return 0 as *mut libc::c_void;
}
#[no_mangle]
pub unsafe extern "C" fn hash_table_remove(
    mut hash_table: *mut HashTable,
    mut key: HashTableKey,
) -> libc::c_int {
    let mut rover: *mut *mut HashTableEntry = 0 as *mut *mut HashTableEntry;
    let mut entry: *mut HashTableEntry = 0 as *mut HashTableEntry;
    let mut pair: *mut HashTablePair = 0 as *mut HashTablePair;
    let mut index: libc::c_uint = 0;
    let mut result: libc::c_int = 0;
    index = (((*hash_table).hash_func).expect("non-null function pointer")(key))
        .wrapping_rem((*hash_table).table_size);
    result = 0 as libc::c_int;
    rover = &mut *((*hash_table).table).offset(index as isize)
        as *mut *mut HashTableEntry;
    while !(*rover).is_null() {
        pair = &mut (**rover).pair;
        if ((*hash_table).equal_func)
            .expect("non-null function pointer")(key, (*pair).key) != 0 as libc::c_int
        {
            entry = *rover;
            *rover = (*entry).next;
            hash_table_free_entry(hash_table, entry);
            (*hash_table).entries = ((*hash_table).entries).wrapping_sub(1);
            (*hash_table).entries;
            result = 1 as libc::c_int;
            break;
        } else {
            rover = &mut (**rover).next;
        }
    }
    return result;
}
#[no_mangle]
pub unsafe extern "C" fn hash_table_num_entries(
    mut hash_table: *mut HashTable,
) -> libc::c_uint {
    return (*hash_table).entries;
}
#[no_mangle]
pub unsafe extern "C" fn hash_table_iterate(
    mut hash_table: *mut HashTable,
    mut iterator: *mut HashTableIterator,
) {
    let mut chain: libc::c_uint = 0;
    (*iterator).hash_table = hash_table;
    (*iterator).next_entry = 0 as *mut HashTableEntry;
    chain = 0 as libc::c_int as libc::c_uint;
    while chain < (*hash_table).table_size {
        if !(*((*hash_table).table).offset(chain as isize)).is_null() {
            (*iterator).next_entry = *((*hash_table).table).offset(chain as isize);
            (*iterator).next_chain = chain;
            break;
        } else {
            chain = chain.wrapping_add(1);
            chain;
        }
    }
}
#[no_mangle]
pub unsafe extern "C" fn hash_table_iter_has_more(
    mut iterator: *mut HashTableIterator,
) -> libc::c_int {
    return ((*iterator).next_entry != 0 as *mut libc::c_void as *mut HashTableEntry)
        as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn hash_table_iter_next(
    mut iterator: *mut HashTableIterator,
) -> HashTablePair {
    let mut current_entry: *mut HashTableEntry = 0 as *mut HashTableEntry;
    let mut hash_table: *mut HashTable = 0 as *mut HashTable;
    let mut pair: HashTablePair = {
        let mut init = _HashTablePair {
            key: 0 as *mut libc::c_void,
            value: 0 as *mut libc::c_void,
        };
        init
    };
    let mut chain: libc::c_uint = 0;
    hash_table = (*iterator).hash_table;
    if ((*iterator).next_entry).is_null() {
        return pair;
    }
    current_entry = (*iterator).next_entry;
    pair = (*current_entry).pair;
    if !((*current_entry).next).is_null() {
        (*iterator).next_entry = (*current_entry).next;
    } else {
        chain = ((*iterator).next_chain).wrapping_add(1 as libc::c_int as libc::c_uint);
        (*iterator).next_entry = 0 as *mut HashTableEntry;
        while chain < (*hash_table).table_size {
            if !(*((*hash_table).table).offset(chain as isize)).is_null() {
                (*iterator).next_entry = *((*hash_table).table).offset(chain as isize);
                break;
            } else {
                chain = chain.wrapping_add(1);
                chain;
            }
        }
        (*iterator).next_chain = chain;
    }
    return pair;
}
unsafe extern "C" fn run_static_initializers() {
    hash_table_num_primes = (::core::mem::size_of::<[libc::c_uint; 24]>()
        as libc::c_ulong)
        .wrapping_div(::core::mem::size_of::<libc::c_int>() as libc::c_ulong)
        as libc::c_uint;
}
#[used]
#[cfg_attr(target_os = "linux", link_section = ".init_array")]
#[cfg_attr(target_os = "windows", link_section = ".CRT$XIB")]
#[cfg_attr(target_os = "macos", link_section = "__DATA,__mod_init_func")]
static INIT_ARRAY: [unsafe extern "C" fn(); 1] = [run_static_initializers];
