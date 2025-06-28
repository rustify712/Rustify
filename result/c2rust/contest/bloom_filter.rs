use ::libc;
extern "C" {
    fn memcpy(
        _: *mut libc::c_void,
        _: *const libc::c_void,
        _: libc::c_ulong,
    ) -> *mut libc::c_void;
    fn alloc_test_malloc(bytes: size_t) -> *mut libc::c_void;
    fn alloc_test_free(ptr: *mut libc::c_void);
    fn alloc_test_calloc(nmemb: size_t, bytes: size_t) -> *mut libc::c_void;
}
pub type size_t = libc::c_ulong;
#[derive(Copy, Clone)]
#[repr(C)]
pub struct _BloomFilter {
    pub hash_func: BloomFilterHashFunc,
    pub table: *mut libc::c_uchar,
    pub table_size: libc::c_uint,
    pub num_functions: libc::c_uint,
}
pub type BloomFilterHashFunc = Option::<
    unsafe extern "C" fn(BloomFilterValue) -> libc::c_uint,
>;
pub type BloomFilterValue = *mut libc::c_void;
pub type BloomFilter = _BloomFilter;
static mut salts: [libc::c_uint; 64] = [
    0x1953c322 as libc::c_int as libc::c_uint,
    0x588ccf17 as libc::c_int as libc::c_uint,
    0x64bf600c as libc::c_int as libc::c_uint,
    0xa6be3f3d as libc::c_uint,
    0x341a02ea as libc::c_int as libc::c_uint,
    0x15b03217 as libc::c_int as libc::c_uint,
    0x3b062858 as libc::c_int as libc::c_uint,
    0x5956fd06 as libc::c_int as libc::c_uint,
    0x18b5624f as libc::c_int as libc::c_uint,
    0xe3be0b46 as libc::c_uint,
    0x20ffcd5c as libc::c_int as libc::c_uint,
    0xa35dfd2b as libc::c_uint,
    0x1fc4a9bf as libc::c_int as libc::c_uint,
    0x57c45d5c as libc::c_int as libc::c_uint,
    0xa8661c4a as libc::c_uint,
    0x4f1b74d2 as libc::c_int as libc::c_uint,
    0x5a6dde13 as libc::c_int as libc::c_uint,
    0x3b18dac6 as libc::c_int as libc::c_uint,
    0x5a8afbf as libc::c_int as libc::c_uint,
    0xbbda2fe2 as libc::c_uint,
    0xa2520d78 as libc::c_uint,
    0xe7934849 as libc::c_uint,
    0xd541bc75 as libc::c_uint,
    0x9a55b57 as libc::c_int as libc::c_uint,
    0x9b345ae2 as libc::c_uint,
    0xfc2d26af as libc::c_uint,
    0x38679cef as libc::c_int as libc::c_uint,
    0x81bd1e0d as libc::c_uint,
    0x654681ae as libc::c_int as libc::c_uint,
    0x4b3d87ad as libc::c_int as libc::c_uint,
    0xd5ff10fb as libc::c_uint,
    0x23b32f67 as libc::c_int as libc::c_uint,
    0xafc7e366 as libc::c_uint,
    0xdd955ead as libc::c_uint,
    0xe7c34b1c as libc::c_uint,
    0xfeace0a6 as libc::c_uint,
    0xeb16f09d as libc::c_uint,
    0x3c57a72d as libc::c_int as libc::c_uint,
    0x2c8294c5 as libc::c_int as libc::c_uint,
    0xba92662a as libc::c_uint,
    0xcd5b2d14 as libc::c_uint,
    0x743936c8 as libc::c_int as libc::c_uint,
    0x2489beff as libc::c_int as libc::c_uint,
    0xc6c56e00 as libc::c_uint,
    0x74a4f606 as libc::c_int as libc::c_uint,
    0xb244a94a as libc::c_uint,
    0x5edfc423 as libc::c_int as libc::c_uint,
    0xf1901934 as libc::c_uint,
    0x24af7691 as libc::c_int as libc::c_uint,
    0xf6c98b25 as libc::c_uint,
    0xea25af46 as libc::c_uint,
    0x76d5f2e6 as libc::c_int as libc::c_uint,
    0x5e33cdf2 as libc::c_int as libc::c_uint,
    0x445eb357 as libc::c_int as libc::c_uint,
    0x88556bd2 as libc::c_uint,
    0x70d1da7a as libc::c_int as libc::c_uint,
    0x54449368 as libc::c_int as libc::c_uint,
    0x381020bc as libc::c_int as libc::c_uint,
    0x1c0520bf as libc::c_int as libc::c_uint,
    0xf7e44942 as libc::c_uint,
    0xa27e2a58 as libc::c_uint,
    0x66866fc5 as libc::c_int as libc::c_uint,
    0x12519ce7 as libc::c_int as libc::c_uint,
    0x437a8456 as libc::c_int as libc::c_uint,
];
#[no_mangle]
pub unsafe extern "C" fn bloom_filter_new(
    mut table_size: libc::c_uint,
    mut hash_func: BloomFilterHashFunc,
    mut num_functions: libc::c_uint,
) -> *mut BloomFilter {
    let mut filter: *mut BloomFilter = 0 as *mut BloomFilter;
    if num_functions as libc::c_ulong
        > (::core::mem::size_of::<[libc::c_uint; 64]>() as libc::c_ulong)
            .wrapping_div(::core::mem::size_of::<libc::c_uint>() as libc::c_ulong)
    {
        return 0 as *mut BloomFilter;
    }
    filter = alloc_test_malloc(::core::mem::size_of::<BloomFilter>() as libc::c_ulong)
        as *mut BloomFilter;
    if filter.is_null() {
        return 0 as *mut BloomFilter;
    }
    (*filter)
        .table = alloc_test_calloc(
        table_size
            .wrapping_add(7 as libc::c_int as libc::c_uint)
            .wrapping_div(8 as libc::c_int as libc::c_uint) as size_t,
        1 as libc::c_int as size_t,
    ) as *mut libc::c_uchar;
    if ((*filter).table).is_null() {
        alloc_test_free(filter as *mut libc::c_void);
        return 0 as *mut BloomFilter;
    }
    (*filter).hash_func = hash_func;
    (*filter).num_functions = num_functions;
    (*filter).table_size = table_size;
    return filter;
}
#[no_mangle]
pub unsafe extern "C" fn bloom_filter_free(mut bloomfilter: *mut BloomFilter) {
    alloc_test_free((*bloomfilter).table as *mut libc::c_void);
    alloc_test_free(bloomfilter as *mut libc::c_void);
}
#[no_mangle]
pub unsafe extern "C" fn bloom_filter_insert(
    mut bloomfilter: *mut BloomFilter,
    mut value: BloomFilterValue,
) {
    let mut hash: libc::c_uint = 0;
    let mut subhash: libc::c_uint = 0;
    let mut index: libc::c_uint = 0;
    let mut i: libc::c_uint = 0;
    let mut b: libc::c_uchar = 0;
    hash = ((*bloomfilter).hash_func).expect("non-null function pointer")(value);
    i = 0 as libc::c_int as libc::c_uint;
    while i < (*bloomfilter).num_functions {
        subhash = hash ^ salts[i as usize];
        index = subhash.wrapping_rem((*bloomfilter).table_size);
        b = ((1 as libc::c_int) << index.wrapping_rem(8 as libc::c_int as libc::c_uint))
            as libc::c_uchar;
        let ref mut fresh0 = *((*bloomfilter).table)
            .offset(index.wrapping_div(8 as libc::c_int as libc::c_uint) as isize);
        *fresh0 = (*fresh0 as libc::c_int | b as libc::c_int) as libc::c_uchar;
        i = i.wrapping_add(1);
        i;
    }
}
#[no_mangle]
pub unsafe extern "C" fn bloom_filter_query(
    mut bloomfilter: *mut BloomFilter,
    mut value: BloomFilterValue,
) -> libc::c_int {
    let mut hash: libc::c_uint = 0;
    let mut subhash: libc::c_uint = 0;
    let mut index: libc::c_uint = 0;
    let mut i: libc::c_uint = 0;
    let mut b: libc::c_uchar = 0;
    let mut bit: libc::c_int = 0;
    hash = ((*bloomfilter).hash_func).expect("non-null function pointer")(value);
    i = 0 as libc::c_int as libc::c_uint;
    while i < (*bloomfilter).num_functions {
        subhash = hash ^ salts[i as usize];
        index = subhash.wrapping_rem((*bloomfilter).table_size);
        b = *((*bloomfilter).table)
            .offset(index.wrapping_div(8 as libc::c_int as libc::c_uint) as isize);
        bit = (1 as libc::c_int) << index.wrapping_rem(8 as libc::c_int as libc::c_uint);
        if b as libc::c_int & bit == 0 as libc::c_int {
            return 0 as libc::c_int;
        }
        i = i.wrapping_add(1);
        i;
    }
    return 1 as libc::c_int;
}
#[no_mangle]
pub unsafe extern "C" fn bloom_filter_read(
    mut bloomfilter: *mut BloomFilter,
    mut array: *mut libc::c_uchar,
) {
    let mut array_size: libc::c_uint = 0;
    array_size = ((*bloomfilter).table_size)
        .wrapping_add(7 as libc::c_int as libc::c_uint)
        .wrapping_div(8 as libc::c_int as libc::c_uint);
    memcpy(
        array as *mut libc::c_void,
        (*bloomfilter).table as *const libc::c_void,
        array_size as libc::c_ulong,
    );
}
#[no_mangle]
pub unsafe extern "C" fn bloom_filter_load(
    mut bloomfilter: *mut BloomFilter,
    mut array: *mut libc::c_uchar,
) {
    let mut array_size: libc::c_uint = 0;
    array_size = ((*bloomfilter).table_size)
        .wrapping_add(7 as libc::c_int as libc::c_uint)
        .wrapping_div(8 as libc::c_int as libc::c_uint);
    memcpy(
        (*bloomfilter).table as *mut libc::c_void,
        array as *const libc::c_void,
        array_size as libc::c_ulong,
    );
}
#[no_mangle]
pub unsafe extern "C" fn bloom_filter_union(
    mut filter1: *mut BloomFilter,
    mut filter2: *mut BloomFilter,
) -> *mut BloomFilter {
    let mut result: *mut BloomFilter = 0 as *mut BloomFilter;
    let mut i: libc::c_uint = 0;
    let mut array_size: libc::c_uint = 0;
    if (*filter1).table_size != (*filter2).table_size
        || (*filter1).num_functions != (*filter2).num_functions
        || (*filter1).hash_func != (*filter2).hash_func
    {
        return 0 as *mut BloomFilter;
    }
    result = bloom_filter_new(
        (*filter1).table_size,
        (*filter1).hash_func,
        (*filter1).num_functions,
    );
    if result.is_null() {
        return 0 as *mut BloomFilter;
    }
    array_size = ((*filter1).table_size)
        .wrapping_add(7 as libc::c_int as libc::c_uint)
        .wrapping_div(8 as libc::c_int as libc::c_uint);
    i = 0 as libc::c_int as libc::c_uint;
    while i < array_size {
        *((*result).table)
            .offset(
                i as isize,
            ) = (*((*filter1).table).offset(i as isize) as libc::c_int
            | *((*filter2).table).offset(i as isize) as libc::c_int) as libc::c_uchar;
        i = i.wrapping_add(1);
        i;
    }
    return result;
}
#[no_mangle]
pub unsafe extern "C" fn bloom_filter_intersection(
    mut filter1: *mut BloomFilter,
    mut filter2: *mut BloomFilter,
) -> *mut BloomFilter {
    let mut result: *mut BloomFilter = 0 as *mut BloomFilter;
    let mut i: libc::c_uint = 0;
    let mut array_size: libc::c_uint = 0;
    if (*filter1).table_size != (*filter2).table_size
        || (*filter1).num_functions != (*filter2).num_functions
        || (*filter1).hash_func != (*filter2).hash_func
    {
        return 0 as *mut BloomFilter;
    }
    result = bloom_filter_new(
        (*filter1).table_size,
        (*filter1).hash_func,
        (*filter1).num_functions,
    );
    if result.is_null() {
        return 0 as *mut BloomFilter;
    }
    array_size = ((*filter1).table_size)
        .wrapping_add(7 as libc::c_int as libc::c_uint)
        .wrapping_div(8 as libc::c_int as libc::c_uint);
    i = 0 as libc::c_int as libc::c_uint;
    while i < array_size {
        *((*result).table)
            .offset(
                i as isize,
            ) = (*((*filter1).table).offset(i as isize) as libc::c_int
            & *((*filter2).table).offset(i as isize) as libc::c_int) as libc::c_uchar;
        i = i.wrapping_add(1);
        i;
    }
    return result;
}
