from transformers import AutoTokenizer, PreTrainedTokenizer

TOKENIZER_PATH_DICT = {
    "deepseek-chat": "core/tokenizer/deepseek_v2"
}


class Tokenizer:

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self._tokenizer = tokenizer

    @property
    def tokenizer(self):
        return self._tokenizer

    @staticmethod
    def from_pretrained(pretrained_model_name_or_path: str, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return Tokenizer(tokenizer)

    def text_token_count(self, text: str):
        """计算文本的 token 数量"""
        return len(self._tokenizer.encode(text))

    def messages_token_count(self, messages: list[dict]):
        """计算消息的 token 数量"""
        input_ids = self._tokenizer.apply_chat_template(messages)
        return len(input_ids)


class TokenizerManager:
    tokenizer_dict = {}

    @classmethod
    def get_tokenizer(cls, tokenizer_name: str):
        if tokenizer_name not in TOKENIZER_PATH_DICT:
            raise ValueError(f"Unknown tokenizer: {tokenizer_name}")
        if tokenizer_name in cls.tokenizer_dict:
            return cls.tokenizer_dict[tokenizer_name]
        tokenizer_path = TOKENIZER_PATH_DICT[tokenizer_name]
        tokenizer = Tokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        cls.tokenizer_dict[tokenizer_name] = tokenizer
        return tokenizer

if __name__ == '__main__':
    tokenizer = TokenizerManager.get_tokenizer("deepseek-chat")
    code = r"""
You are adept at translating C/C++ code to Rust with high accuracy, ensuring that all syntax, semantics, and specific language features are correctly and efficiently converted.
The goal is to produce idiomatic, safe, and efficient Rust code with a focus on memory safety and modern Rust practices.
Every function, interface, and implementation must be accurately represented in Rust.Translate while strictly following the rules below:
- Do not show `Examples` in Rust docstrings.
- Use only the following external Rust crates: rand="0.4", regex="1", and md5="0.7.0".
- use `Vec` instead of arrays for dynamic collections.
- use `String` instead of `&str` for owned and mutable string data.
- Do not define an entry point (e.g., main) or testing functions.
- use `Result` or `Option` for error handling.
- Translate each C/C++ function or interface as an equivalent Rust function or method with all input/output parameters and return values adapted to Rust idioms.
- Ensure proper memory safety by replacing manual memory management (malloc, free) with Rust's ownership and borrowing model.
- Replace macros with idiomatic Rust constructs, such as const, static, or generic functions, depending on the context.
- Inline comments from the C/C++ code must be preserved as Rust comments for clarity.
- Handle unsafe operations (e.g., pointer arithmetic) explicitly using Rust's unsafe block while minimizing its use.
- Avoid unsafe code and raw pointers unless strictly necessary. When unavoidable, unsafe blocks must be explicitly documented and minimized, with clear explanations for their necessity.Translate the following C/C++ code to Rust:

// TO ENABLE INLINE FUNCTIONS:
//   ON MSVC: enable the 'Inline Function Expansion' (/Ob2) compiler option, and maybe the
//            'Whole Program Optimitazion' (/GL), that requires the
//            'Link Time Code Generation' (/LTCG) linker option to be enabled too

#ifndef BINN_H
#define BINN_H

#ifdef __cplusplus
extern "C" {
#endif

#define BINN_VERSION "3.0.0"  /* using semantic versioning */

#ifndef NULL
#ifdef __cplusplus
#define NULL    0
#else
#define NULL    ((void *)0)
#endif
#endif

#ifndef TRUE
#define TRUE  1
#endif

#ifndef FALSE
#define FALSE 0
#endif

#ifndef BOOL
typedef int BOOL;
#endif

#ifndef APIENTRY
 #ifdef _WIN32
  #define APIENTRY __stdcall
 #else
  //#define APIENTRY __attribute__((stdcall))
  #define APIENTRY 
 #endif
#endif

#ifndef BINN_PRIVATE
 #ifdef DEBUG
  #define BINN_PRIVATE
 #else
  #define BINN_PRIVATE  static
 #endif
#endif

#ifdef _MSC_VER
  #define INLINE         __inline
  #define ALWAYS_INLINE  __forceinline
#else
  // you can change to 'extern inline' if using the gcc option -flto
  #define INLINE         static inline
  #define ALWAYS_INLINE  static inline __attribute__((always_inline))
#endif

#ifndef int64
#if defined(_MSC_VER) || defined(__BORLANDC__)
  typedef __int64 int64;
  typedef unsigned __int64 uint64;
#else
  typedef long long int int64;
  typedef unsigned long long int uint64;
#endif
#endif

#ifdef _WIN32
#define INT64_FORMAT  "I64i"
#define UINT64_FORMAT "I64u"
#define INT64_HEX_FORMAT  "I64x"
#else
#define INT64_FORMAT  "lli"
#define UINT64_FORMAT "llu"
#define INT64_HEX_FORMAT  "llx"
#endif


// BINN CONSTANTS  ----------------------------------------

#define INVALID_BINN         0

// Storage Data Types  ------------------------------------

#define BINN_STORAGE_NOBYTES   0x00
#define BINN_STORAGE_BYTE      0x20  //  8 bits
#define BINN_STORAGE_WORD      0x40  // 16 bits -- the endianess (byte order) is automatically corrected
#define BINN_STORAGE_DWORD     0x60  // 32 bits -- the endianess (byte order) is automatically corrected
#define BINN_STORAGE_QWORD     0x80  // 64 bits -- the endianess (byte order) is automatically corrected
#define BINN_STORAGE_STRING    0xA0  // Are stored with null termination
#define BINN_STORAGE_BLOB      0xC0
#define BINN_STORAGE_CONTAINER 0xE0
#define BINN_STORAGE_VIRTUAL   0x80000

#define BINN_STORAGE_MIN       BINN_STORAGE_NOBYTES
#define BINN_STORAGE_MAX       BINN_STORAGE_CONTAINER

#define BINN_STORAGE_MASK      0xE0
#define BINN_STORAGE_MASK16    0xE000
#define BINN_STORAGE_HAS_MORE  0x10
#define BINN_TYPE_MASK         0x0F
#define BINN_TYPE_MASK16       0x0FFF

#define BINN_MAX_VALUE_MASK    0xFFFFF


// Data Formats  ------------------------------------------

#define BINN_LIST      0xE0
#define BINN_MAP       0xE1
#define BINN_OBJECT    0xE2

#define BINN_NULL      0x00
#define BINN_TRUE      0x01
#define BINN_FALSE     0x02

#define BINN_UINT8     0x20  // (BYTE) (unsigned byte) Is the default format for the BYTE type
#define BINN_INT8      0x21  // (BYTE) (signed byte, from -128 to +127. The 0x80 is the sign bit, so the range in hex is from 0x80 [-128] to 0x7F [127], being 0x00 = 0 and 0xFF = -1)
#define BINN_UINT16    0x40  // (WORD) (unsigned integer) Is the default format for the WORD type
#define BINN_INT16     0x41  // (WORD) (signed integer)
#define BINN_UINT32    0x60  // (DWORD) (unsigned integer) Is the default format for the DWORD type
#define BINN_INT32     0x61  // (DWORD) (signed integer)
#define BINN_UINT64    0x80  // (QWORD) (unsigned integer) Is the default format for the QWORD type
#define BINN_INT64     0x81  // (QWORD) (signed integer)

#define BINN_SCHAR     BINN_INT8
#define BINN_UCHAR     BINN_UINT8

#define BINN_STRING    0xA0  // (STRING) Raw String
#define BINN_DATETIME  0xA1  // (STRING) iso8601 format -- YYYY-MM-DD HH:MM:SS
#define BINN_DATE      0xA2  // (STRING) iso8601 format -- YYYY-MM-DD
#define BINN_TIME      0xA3  // (STRING) iso8601 format -- HH:MM:SS
#define BINN_DECIMAL   0xA4  // (STRING) High precision number - used for generic decimal values and for those ones that cannot be represented in the float64 format.
#define BINN_CURRENCYSTR  0xA5  // (STRING) With currency unit/symbol - check for some iso standard format
#define BINN_SINGLE_STR   0xA6  // (STRING) Can be restored to float32
#define BINN_DOUBLE_STR   0xA7  // (STRING) May be restored to float64

#define BINN_FLOAT32   0x62  // (DWORD) 
#define BINN_FLOAT64   0x82  // (QWORD) 
#define BINN_FLOAT     BINN_FLOAT32
#define BINN_SINGLE    BINN_FLOAT32
#define BINN_DOUBLE    BINN_FLOAT64

#define BINN_CURRENCY  0x83  // (QWORD)

#define BINN_BLOB      0xC0  // (BLOB) Raw Blob


// virtual types:

#define BINN_BOOL      0x80061  // (DWORD) The value may be 0 or 1

#ifdef BINN_EXTENDED
//#define BINN_SINGLE    0x800A1  // (STRING) Can be restored to float32
//#define BINN_DOUBLE    0x800A2  // (STRING) May be restored to float64
#endif

//#define BINN_BINN      0x800E1  // (CONTAINER)
//#define BINN_BINN_BUFFER  0x800C1  // (BLOB) user binn. it's not open by the parser


// extended content types:

// strings:

#define BINN_HTML      0xB001
#define BINN_XML       0xB002
#define BINN_JSON      0xB003
#define BINN_JAVASCRIPT 0xB004
#define BINN_CSS       0xB005

// blobs:

#define BINN_JPEG      0xD001
#define BINN_GIF       0xD002
#define BINN_PNG       0xD003
#define BINN_BMP       0xD004


// type families
#define BINN_FAMILY_NONE   0x00
#define BINN_FAMILY_NULL   0xf1
#define BINN_FAMILY_INT    0xf2
#define BINN_FAMILY_FLOAT  0xf3
#define BINN_FAMILY_STRING 0xf4
#define BINN_FAMILY_BLOB   0xf5
#define BINN_FAMILY_BOOL   0xf6
#define BINN_FAMILY_BINN   0xf7

// integer types related to signal
#define BINN_SIGNED_INT     11
#define BINN_UNSIGNED_INT   22


typedef void (*binn_mem_free)(void*);
#define BINN_STATIC      ((binn_mem_free)0)
#define BINN_TRANSIENT   ((binn_mem_free)-1)


// --- BINN STRUCTURE --------------------------------------------------------------


struct binn_struct {
  int    header;     // this struct header holds the magic number (BINN_MAGIC) that identifies this memory block as a binn structure
  BOOL   allocated;  // the struct can be allocated using malloc_fn() or can be on the stack
  BOOL   writable;   // did it was create for writing? it can use the pbuf if not unified with ptr
  BOOL   dirty;      // the container header is not written to the buffer
  //
  void  *pbuf;       // use *ptr below?
  BOOL   pre_allocated;
  int    alloc_size;
  int    used_size;
  //
  int    type;
  void  *ptr;
  int    size;
  int    count;
  //
  binn_mem_free freefn;  // used only when type == BINN_STRING or BINN_BLOB
  //
  union {
    signed char    vint8;
    signed short   vint16;
    signed int     vint32;
    int64          vint64;
    unsigned char  vuint8;
    unsigned short vuint16;
    unsigned int   vuint32;
    uint64         vuint64;
    //
    signed char    vchar;
    unsigned char  vuchar;
    signed short   vshort;
    unsigned short vushort;
    signed int     vint;
    unsigned int   vuint;
    //
    float          vfloat;
    double         vdouble;
    //
    BOOL           vbool;
  };
  //
  BOOL   disable_int_compression;
};

typedef struct binn_struct binn;



// --- GENERAL FUNCTIONS  ----------------------------------------------------------

char * APIENTRY binn_version();

void   APIENTRY binn_set_alloc_functions(void* (*new_malloc)(size_t), void* (*new_realloc)(void*,size_t), void (*new_free)(void*));

int    APIENTRY binn_create_type(int storage_type, int data_type_index);
BOOL   APIENTRY binn_get_type_info(int long_type, int *pstorage_type, int *pextra_type);

int    APIENTRY binn_get_write_storage(int type);
int    APIENTRY binn_get_read_storage(int type);

BOOL   APIENTRY binn_is_container(binn *item);


// --- WRITE FUNCTIONS  ------------------------------------------------------------

// create a new binn allocating memory for the structure
binn * APIENTRY binn_new(int type, int size, void *buffer);
binn * APIENTRY binn_list();
binn * APIENTRY binn_map();
binn * APIENTRY binn_object();

// create a new binn storing the structure on the stack
BOOL APIENTRY binn_create(binn *item, int type, int size, void *buffer);
BOOL APIENTRY binn_create_list(binn *list);
BOOL APIENTRY binn_create_map(binn *map);
BOOL APIENTRY binn_create_object(binn *object);

// create a new binn as a copy from another
binn * APIENTRY binn_copy(void *old);


BOOL APIENTRY binn_list_add_new(binn *list, binn *value);
BOOL APIENTRY binn_map_set_new(binn *map, int id, binn *value);
BOOL APIENTRY binn_object_set_new(binn *obj, const char *key, binn *value);


// extended interface

BOOL   APIENTRY binn_list_add(binn *list, int type, void *pvalue, int size);
BOOL   APIENTRY binn_map_set(binn *map, int id, int type, void *pvalue, int size);
BOOL   APIENTRY binn_object_set(binn *obj, const char *key, int type, void *pvalue, int size);


// release memory

void   APIENTRY binn_free(binn *item);
void * APIENTRY binn_release(binn *item); // free the binn structure but keeps the binn buffer allocated, returning a pointer to it. use the free function to release the buffer later


// --- CREATING VALUES ---------------------------------------------------

binn * APIENTRY binn_value(int type, void *pvalue, int size, binn_mem_free freefn);

ALWAYS_INLINE binn * binn_int8(signed char value) {
  return binn_value(BINN_INT8, &value, 0, NULL);
}
ALWAYS_INLINE binn * binn_int16(short value) {
  return binn_value(BINN_INT16, &value, 0, NULL);
}
ALWAYS_INLINE binn * binn_int32(int value) {
  return binn_value(BINN_INT32, &value, 0, NULL);
}
ALWAYS_INLINE binn * binn_int64(int64 value) {
  return binn_value(BINN_INT64, &value, 0, NULL);
}
ALWAYS_INLINE binn * binn_uint8(unsigned char value) {
  return binn_value(BINN_UINT8, &value, 0, NULL);
}
ALWAYS_INLINE binn * binn_uint16(unsigned short value) {
  return binn_value(BINN_UINT16, &value, 0, NULL);
}
ALWAYS_INLINE binn * binn_uint32(unsigned int value) {
  return binn_value(BINN_UINT32, &value, 0, NULL);
}
ALWAYS_INLINE binn * binn_uint64(uint64 value) {
  return binn_value(BINN_UINT64, &value, 0, NULL);
}
ALWAYS_INLINE binn * binn_float(float value) {
  return binn_value(BINN_FLOAT, &value, 0, NULL);
}
ALWAYS_INLINE binn * binn_double(double value) {
  return binn_value(BINN_DOUBLE, &value, 0, NULL);
}
ALWAYS_INLINE binn * binn_bool(BOOL value) {
  return binn_value(BINN_BOOL, &value, 0, NULL);
}
ALWAYS_INLINE binn * binn_null() {
  return binn_value(BINN_NULL, NULL, 0, NULL);
}
ALWAYS_INLINE binn * binn_string(char *str, binn_mem_free freefn) {
  return binn_value(BINN_STRING, str, 0, freefn);
}
ALWAYS_INLINE binn * binn_blob(void *ptr, int size, binn_mem_free freefn) {
  return binn_value(BINN_BLOB, ptr, size, freefn);
}


// --- READ FUNCTIONS  -------------------------------------------------------------

// these functions accept pointer to the binn structure and pointer to the binn buffer
void * APIENTRY binn_ptr(void *ptr);
int    APIENTRY binn_size(void *ptr);
int    APIENTRY binn_type(void *ptr);
int    APIENTRY binn_count(void *ptr);

BOOL   APIENTRY binn_is_valid(void *ptr, int *ptype, int *pcount, int *psize);
/* the function returns the values (type, count and size) and they don't need to be
   initialized. these values are read from the buffer. example:

   int type, count, size;
   result = binn_is_valid(ptr, &type, &count, &size);
*/
BOOL   APIENTRY binn_is_valid_ex(void *ptr, int *ptype, int *pcount, int *psize);
/* if some value is informed (type, count or size) then the function will check if 
   the value returned from the serialized data matches the informed value. otherwise
   the values must be initialized to zero. example:

   int type=0, count=0, size = known_size;
   result = binn_is_valid_ex(ptr, &type, &count, &size);
*/

BOOL   APIENTRY binn_is_struct(void *ptr);


// Loading a binn buffer into a binn value - this is optional

BOOL   APIENTRY binn_load(void *data, binn *item);  // on stack
binn * APIENTRY binn_open(void *data);              // allocated


// easiest interface to use, but don't check if the value is there

signed char    APIENTRY binn_list_int8(void *list, int pos);
short          APIENTRY binn_list_int16(void *list, int pos);
int            APIENTRY binn_list_int32(void *list, int pos);
int64          APIENTRY binn_list_int64(void *list, int pos);
unsigned char  APIENTRY binn_list_uint8(void *list, int pos);
unsigned short APIENTRY binn_list_uint16(void *list, int pos);
unsigned int   APIENTRY binn_list_uint32(void *list, int pos);
uint64         APIENTRY binn_list_uint64(void *list, int pos);
float          APIENTRY binn_list_float(void *list, int pos);
double         APIENTRY binn_list_double(void *list, int pos);
BOOL           APIENTRY binn_list_bool(void *list, int pos);
BOOL           APIENTRY binn_list_null(void *list, int pos);
char *         APIENTRY binn_list_str(void *list, int pos);
void *         APIENTRY binn_list_blob(void *list, int pos, int *psize);
void *         APIENTRY binn_list_list(void *list, int pos);
void *         APIENTRY binn_list_map(void *list, int pos);
void *         APIENTRY binn_list_object(void *list, int pos);

signed char    APIENTRY binn_map_int8(void *map, int id);
short          APIENTRY binn_map_int16(void *map, int id);
int            APIENTRY binn_map_int32(void *map, int id);
int64          APIENTRY binn_map_int64(void *map, int id);
unsigned char  APIENTRY binn_map_uint8(void *map, int id);
unsigned short APIENTRY binn_map_uint16(void *map, int id);
unsigned int   APIENTRY binn_map_uint32(void *map, int id);
uint64         APIENTRY binn_map_uint64(void *map, int id);
float          APIENTRY binn_map_float(void *map, int id);
double         APIENTRY binn_map_double(void *map, int id);
BOOL           APIENTRY binn_map_bool(void *map, int id);
BOOL           APIENTRY binn_map_null(void *map, int id);
char *         APIENTRY binn_map_str(void *map, int id);
void *         APIENTRY binn_map_blob(void *map, int id, int *psize);
void *         APIENTRY binn_map_list(void *map, int id);
void *         APIENTRY binn_map_map(void *map, int id);
void *         APIENTRY binn_map_object(void *map, int id);

signed char    APIENTRY binn_object_int8(void *obj, const char *key);
short          APIENTRY binn_object_int16(void *obj, const char *key);
int            APIENTRY binn_object_int32(void *obj, const char *key);
int64          APIENTRY binn_object_int64(void *obj, const char *key);
unsigned char  APIENTRY binn_object_uint8(void *obj, const char *key);
unsigned short APIENTRY binn_object_uint16(void *obj, const char *key);
unsigned int   APIENTRY binn_object_uint32(void *obj, const char *key);
uint64         APIENTRY binn_object_uint64(void *obj, const char *key);
float          APIENTRY binn_object_float(void *obj, const char *key);
double         APIENTRY binn_object_double(void *obj, const char *key);
BOOL           APIENTRY binn_object_bool(void *obj, const char *key);
BOOL           APIENTRY binn_object_null(void *obj, const char *key);
char *         APIENTRY binn_object_str(void *obj, const char *key);
void *         APIENTRY binn_object_blob(void *obj, const char *key, int *psize);
void *         APIENTRY binn_object_list(void *obj, const char *key);
void *         APIENTRY binn_object_map(void *obj, const char *key);
void *         APIENTRY binn_object_object(void *obj, const char *key);


// return a pointer to an allocated binn structure - must be released with the free() function or equivalent set in binn_set_alloc_functions()
binn * APIENTRY binn_list_value(void *list, int pos);
binn * APIENTRY binn_map_value(void *map, int id);
binn * APIENTRY binn_object_value(void *obj, const char *key);

// read the value to a binn structure on the stack
BOOL APIENTRY binn_list_get_value(void* list, int pos, binn *value);
BOOL APIENTRY binn_map_get_value(void* map, int id, binn *value);
BOOL APIENTRY binn_object_get_value(void *obj, const char *key, binn *value);

// single interface - these functions check the data type
BOOL APIENTRY binn_list_get(void *list, int pos, int type, void *pvalue, int *psize);
BOOL APIENTRY binn_map_get(void *map, int id, int type, void *pvalue, int *psize);
BOOL APIENTRY binn_object_get(void *obj, const char *key, int type, void *pvalue, int *psize);

// these 3 functions return a pointer to the value and the data type
// they are thread-safe on big-endian devices
// on little-endian devices they are thread-safe only to return pointers to list, map, object, blob and strings
// the returned pointer to 16, 32 and 64 bits values must be used only by single-threaded applications
void * APIENTRY binn_list_read(void *list, int pos, int *ptype, int *psize);
void * APIENTRY binn_map_read(void *map, int id, int *ptype, int *psize);
void * APIENTRY binn_object_read(void *obj, const char *key, int *ptype, int *psize);


// READ PAIR FUNCTIONS

// these functions use base 1 in the 'pos' argument

// on stack
BOOL APIENTRY binn_map_get_pair(void *map, int pos, int *pid, binn *value);
BOOL APIENTRY binn_object_get_pair(void *obj, int pos, char *pkey, binn *value);  // must free the memory returned in the pkey

// allocated
binn * APIENTRY binn_map_pair(void *map, int pos, int *pid);
binn * APIENTRY binn_object_pair(void *obj, int pos, char *pkey);  // must free the memory returned in the pkey

// these 2 functions return a pointer to the value and the data type
// they are thread-safe on big-endian devices
// on little-endian devices they are thread-safe only to return pointers to list, map, object, blob and strings
// the returned pointer to 16, 32 and 64 bits values must be used only by single-threaded applications
void * APIENTRY binn_map_read_pair(void *ptr, int pos, int *pid, int *ptype, int *psize);
void * APIENTRY binn_object_read_pair(void *ptr, int pos, char *pkey, int *ptype, int *psize);


// SEQUENTIAL READ FUNCTIONS

typedef struct binn_iter_struct {
    unsigned char *pnext;
    unsigned char *plimit;
    int   type;
    int   count;
    int   current;
} binn_iter;

BOOL   APIENTRY binn_iter_init(binn_iter *iter, void *pbuf, int type);

// allocated
binn * APIENTRY binn_list_next_value(binn_iter *iter);
binn * APIENTRY binn_map_next_value(binn_iter *iter, int *pid);
binn * APIENTRY binn_object_next_value(binn_iter *iter, char *pkey);  // the key must be declared as: char key[256];

// on stack
BOOL   APIENTRY binn_list_next(binn_iter *iter, binn *value);
BOOL   APIENTRY binn_map_next(binn_iter *iter, int *pid, binn *value);
BOOL   APIENTRY binn_object_next(binn_iter *iter, char *pkey, binn *value);  // the key must be declared as: char key[256];

// these 3 functions return a pointer to the value and the data type
// they are thread-safe on big-endian devices
// on little-endian devices they are thread-safe only to return pointers to list, map, object, blob and strings
// the returned pointer to 16, 32 and 64 bits values must be used only by single-threaded applications
void * APIENTRY binn_list_read_next(binn_iter *iter, int *ptype, int *psize);
void * APIENTRY binn_map_read_next(binn_iter *iter, int *pid, int *ptype, int *psize);
void * APIENTRY binn_object_read_next(binn_iter *iter, char *pkey, int *ptype, int *psize);  // the key must be declared as: char key[256];


// --- MACROS ------------------------------------------------------------


#define binn_is_writable(item) (item)->writable;


// set values on stack allocated binn structures

#define binn_set_null(item)         do { (item)->type = BINN_NULL; } while (0)

#define binn_set_bool(item,value)   do { (item)->type = BINN_BOOL; (item)->vbool = value; (item)->ptr = &((item)->vbool); } while (0)

#define binn_set_int(item,value)    do { (item)->type = BINN_INT32; (item)->vint32 = value; (item)->ptr = &((item)->vint32); } while (0)
#define binn_set_int64(item,value)  do { (item)->type = BINN_INT64; (item)->vint64 = value; (item)->ptr = &((item)->vint64); } while (0)

#define binn_set_uint(item,value)   do { (item)->type = BINN_UINT32; (item)->vuint32 = value; (item)->ptr = &((item)->vuint32); } while (0)
#define binn_set_uint64(item,value) do { (item)->type = BINN_UINT64; (item)->vuint64 = value; (item)->ptr = &((item)->vuint64); } while (0)

#define binn_set_float(item,value)  do { (item)->type = BINN_FLOAT;  (item)->vfloat  = value; (item)->ptr = &((item)->vfloat); } while (0)
#define binn_set_double(item,value) do { (item)->type = BINN_DOUBLE; (item)->vdouble = value; (item)->ptr = &((item)->vdouble); } while (0)

//#define binn_set_string(item,str,pfree)    do { (item)->type = BINN_STRING; (item)->ptr = str; (item)->freefn = pfree; } while (0)
//#define binn_set_blob(item,ptr,size,pfree) do { (item)->type = BINN_BLOB;   (item)->ptr = ptr; (item)->freefn = pfree; (item)->size = size; } while (0)
BOOL APIENTRY binn_set_string(binn *item, char *str, binn_mem_free pfree);
BOOL APIENTRY binn_set_blob(binn *item, void *ptr, int size, binn_mem_free pfree);


//#define binn_double(value) {       (item)->type = BINN_DOUBLE; (item)->vdouble = value; (item)->ptr = &((item)->vdouble) }



// FOREACH MACROS
// must use these declarations in the function that will use them:
//  binn_iter iter;
//  char key[256];  // only for the object
//  int  id;        // only for the map
//  binn value;

#define binn_object_foreach(object, key, value)   \
    binn_iter_init(&iter, object, BINN_OBJECT);   \
    while (binn_object_next(&iter, key, &value))

#define binn_map_foreach(map, id, value)      \
    binn_iter_init(&iter, map, BINN_MAP);     \
    while (binn_map_next(&iter, &id, &value))

#define binn_list_foreach(list, value)      \
    binn_iter_init(&iter, list, BINN_LIST); \
    while (binn_list_next(&iter, &value))



/*************************************************************************************/
/*** SET FUNCTIONS *******************************************************************/
/*************************************************************************************/

ALWAYS_INLINE BOOL binn_list_add_int8(binn *list, signed char value) {
  return binn_list_add(list, BINN_INT8, &value, 0);
}
ALWAYS_INLINE BOOL binn_list_add_int16(binn *list, short value) {
  return binn_list_add(list, BINN_INT16, &value, 0);
}
ALWAYS_INLINE BOOL binn_list_add_int32(binn *list, int value) {
  return binn_list_add(list, BINN_INT32, &value, 0);
}
ALWAYS_INLINE BOOL binn_list_add_int64(binn *list, int64 value) {
  return binn_list_add(list, BINN_INT64, &value, 0);
}
ALWAYS_INLINE BOOL binn_list_add_uint8(binn *list, unsigned char value) {
  return binn_list_add(list, BINN_UINT8, &value, 0);
}
ALWAYS_INLINE BOOL binn_list_add_uint16(binn *list, unsigned short value) {
  return binn_list_add(list, BINN_UINT16, &value, 0);
}
ALWAYS_INLINE BOOL binn_list_add_uint32(binn *list, unsigned int value) {
  return binn_list_add(list, BINN_UINT32, &value, 0);
}
ALWAYS_INLINE BOOL binn_list_add_uint64(binn *list, uint64 value) {
  return binn_list_add(list, BINN_UINT64, &value, 0);
}
ALWAYS_INLINE BOOL binn_list_add_float(binn *list, float value) {
  return binn_list_add(list, BINN_FLOAT32, &value, 0);
}
ALWAYS_INLINE BOOL binn_list_add_double(binn *list, double value) {
  return binn_list_add(list, BINN_FLOAT64, &value, 0);
}
ALWAYS_INLINE BOOL binn_list_add_bool(binn *list, BOOL value) {
  return binn_list_add(list, BINN_BOOL, &value, 0);
}
ALWAYS_INLINE BOOL binn_list_add_null(binn *list) {
  return binn_list_add(list, BINN_NULL, NULL, 0);
}
ALWAYS_INLINE BOOL binn_list_add_str(binn *list, char *str) {
  return binn_list_add(list, BINN_STRING, str, 0);
}
ALWAYS_INLINE BOOL binn_list_add_blob(binn *list, void *ptr, int size) {
  return binn_list_add(list, BINN_BLOB, ptr, size);
}
ALWAYS_INLINE BOOL binn_list_add_list(binn *list, void *list2) {
  return binn_list_add(list, BINN_LIST, binn_ptr(list2), binn_size(list2));
}
ALWAYS_INLINE BOOL binn_list_add_map(binn *list, void *map) {
  return binn_list_add(list, BINN_MAP, binn_ptr(map), binn_size(map));
}
ALWAYS_INLINE BOOL binn_list_add_object(binn *list, void *obj) {
  return binn_list_add(list, BINN_OBJECT, binn_ptr(obj), binn_size(obj));
}
ALWAYS_INLINE BOOL binn_list_add_value(binn *list, binn *value) {
  return binn_list_add(list, value->type, binn_ptr(value), binn_size(value));
}

/*************************************************************************************/

ALWAYS_INLINE BOOL binn_map_set_int8(binn *map, int id, signed char value) {
  return binn_map_set(map, id, BINN_INT8, &value, 0);
}
ALWAYS_INLINE BOOL binn_map_set_int16(binn *map, int id, short value) {
  return binn_map_set(map, id, BINN_INT16, &value, 0);
}
ALWAYS_INLINE BOOL binn_map_set_int32(binn *map, int id, int value) {
  return binn_map_set(map, id, BINN_INT32, &value, 0);
}
ALWAYS_INLINE BOOL binn_map_set_int64(binn *map, int id, int64 value) {
  return binn_map_set(map, id, BINN_INT64, &value, 0);
}
ALWAYS_INLINE BOOL binn_map_set_uint8(binn *map, int id, unsigned char value) {
  return binn_map_set(map, id, BINN_UINT8, &value, 0);
}
ALWAYS_INLINE BOOL binn_map_set_uint16(binn *map, int id, unsigned short value) {
  return binn_map_set(map, id, BINN_UINT16, &value, 0);
}
ALWAYS_INLINE BOOL binn_map_set_uint32(binn *map, int id, unsigned int value) {
  return binn_map_set(map, id, BINN_UINT32, &value, 0);
}
ALWAYS_INLINE BOOL binn_map_set_uint64(binn *map, int id, uint64 value) {
  return binn_map_set(map, id, BINN_UINT64, &value, 0);
}
ALWAYS_INLINE BOOL binn_map_set_float(binn *map, int id, float value) {
  return binn_map_set(map, id, BINN_FLOAT32, &value, 0);
}
ALWAYS_INLINE BOOL binn_map_set_double(binn *map, int id, double value) {
  return binn_map_set(map, id, BINN_FLOAT64, &value, 0);
}
ALWAYS_INLINE BOOL binn_map_set_bool(binn *map, int id, BOOL value) {
  return binn_map_set(map, id, BINN_BOOL, &value, 0);
}
ALWAYS_INLINE BOOL binn_map_set_null(binn *map, int id) {
  return binn_map_set(map, id, BINN_NULL, NULL, 0);
}
ALWAYS_INLINE BOOL binn_map_set_str(binn *map, int id, char *str) {
  return binn_map_set(map, id, BINN_STRING, str, 0);
}
ALWAYS_INLINE BOOL binn_map_set_blob(binn *map, int id, void *ptr, int size) {
  return binn_map_set(map, id, BINN_BLOB, ptr, size);
}
ALWAYS_INLINE BOOL binn_map_set_list(binn *map, int id, void *list) {
  return binn_map_set(map, id, BINN_LIST, binn_ptr(list), binn_size(list));
}
ALWAYS_INLINE BOOL binn_map_set_map(binn *map, int id, void *map2) {
  return binn_map_set(map, id, BINN_MAP, binn_ptr(map2), binn_size(map2));
}
ALWAYS_INLINE BOOL binn_map_set_object(binn *map, int id, void *obj) {
  return binn_map_set(map, id, BINN_OBJECT, binn_ptr(obj), binn_size(obj));
}
ALWAYS_INLINE BOOL binn_map_set_value(binn *map, int id, binn *value) {
  return binn_map_set(map, id, value->type, binn_ptr(value), binn_size(value));
}

/*************************************************************************************/

ALWAYS_INLINE BOOL binn_object_set_int8(binn *obj, const char *key, signed char value) {
  return binn_object_set(obj, key, BINN_INT8, &value, 0);
}
ALWAYS_INLINE BOOL binn_object_set_int16(binn *obj, const char *key, short value) {
  return binn_object_set(obj, key, BINN_INT16, &value, 0);
}
ALWAYS_INLINE BOOL binn_object_set_int32(binn *obj, const char *key, int value) {
  return binn_object_set(obj, key, BINN_INT32, &value, 0);
}
ALWAYS_INLINE BOOL binn_object_set_int64(binn *obj, const char *key, int64 value) {
  return binn_object_set(obj, key, BINN_INT64, &value, 0);
}
ALWAYS_INLINE BOOL binn_object_set_uint8(binn *obj, const char *key, unsigned char value) {
  return binn_object_set(obj, key, BINN_UINT8, &value, 0);
}
ALWAYS_INLINE BOOL binn_object_set_uint16(binn *obj, const char *key, unsigned short value) {
  return binn_object_set(obj, key, BINN_UINT16, &value, 0);
}
ALWAYS_INLINE BOOL binn_object_set_uint32(binn *obj, const char *key, unsigned int value) {
  return binn_object_set(obj, key, BINN_UINT32, &value, 0);
}
ALWAYS_INLINE BOOL binn_object_set_uint64(binn *obj, const char *key, uint64 value) {
  return binn_object_set(obj, key, BINN_UINT64, &value, 0);
}
ALWAYS_INLINE BOOL binn_object_set_float(binn *obj, const char *key, float value) {
  return binn_object_set(obj, key, BINN_FLOAT32, &value, 0);
}
ALWAYS_INLINE BOOL binn_object_set_double(binn *obj, const char *key, double value) {
  return binn_object_set(obj, key, BINN_FLOAT64, &value, 0);
}
ALWAYS_INLINE BOOL binn_object_set_bool(binn *obj, const char *key, BOOL value) {
  return binn_object_set(obj, key, BINN_BOOL, &value, 0);
}
ALWAYS_INLINE BOOL binn_object_set_null(binn *obj, const char *key) {
  return binn_object_set(obj, key, BINN_NULL, NULL, 0);
}
ALWAYS_INLINE BOOL binn_object_set_str(binn *obj, const char *key, char *str) {
  return binn_object_set(obj, key, BINN_STRING, str, 0);
}
ALWAYS_INLINE BOOL binn_object_set_blob(binn *obj, const char *key, void *ptr, int size) {
  return binn_object_set(obj, key, BINN_BLOB, ptr, size);
}
ALWAYS_INLINE BOOL binn_object_set_list(binn *obj, const char *key, void *list) {
  return binn_object_set(obj, key, BINN_LIST, binn_ptr(list), binn_size(list));
}
ALWAYS_INLINE BOOL binn_object_set_map(binn *obj, const char *key, void *map) {
  return binn_object_set(obj, key, BINN_MAP, binn_ptr(map), binn_size(map));
}
ALWAYS_INLINE BOOL binn_object_set_object(binn *obj, const char *key, void *obj2) {
  return binn_object_set(obj, key, BINN_OBJECT, binn_ptr(obj2), binn_size(obj2));
}
ALWAYS_INLINE BOOL binn_object_set_value(binn *obj, const char *key, binn *value) {
  return binn_object_set(obj, key, value->type, binn_ptr(value), binn_size(value));
}

/*************************************************************************************/
/*** GET FUNCTIONS *******************************************************************/
/*************************************************************************************/

ALWAYS_INLINE BOOL binn_list_get_int8(void *list, int pos, signed char *pvalue) {
  return binn_list_get(list, pos, BINN_INT8, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_list_get_int16(void *list, int pos, short *pvalue) {
  return binn_list_get(list, pos, BINN_INT16, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_list_get_int32(void *list, int pos, int *pvalue) {
  return binn_list_get(list, pos, BINN_INT32, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_list_get_int64(void *list, int pos, int64 *pvalue) {
  return binn_list_get(list, pos, BINN_INT64, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_list_get_uint8(void *list, int pos, unsigned char *pvalue) {
  return binn_list_get(list, pos, BINN_UINT8, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_list_get_uint16(void *list, int pos, unsigned short *pvalue) {
  return binn_list_get(list, pos, BINN_UINT16, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_list_get_uint32(void *list, int pos, unsigned int *pvalue) {
  return binn_list_get(list, pos, BINN_UINT32, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_list_get_uint64(void *list, int pos, uint64 *pvalue) {
  return binn_list_get(list, pos, BINN_UINT64, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_list_get_float(void *list, int pos, float *pvalue) {
  return binn_list_get(list, pos, BINN_FLOAT32, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_list_get_double(void *list, int pos, double *pvalue) {
  return binn_list_get(list, pos, BINN_FLOAT64, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_list_get_bool(void *list, int pos, BOOL *pvalue) {
  return binn_list_get(list, pos, BINN_BOOL, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_list_get_str(void *list, int pos, char **pvalue) {
  return binn_list_get(list, pos, BINN_STRING, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_list_get_blob(void *list, int pos, void **pvalue, int *psize) {
  return binn_list_get(list, pos, BINN_BLOB, pvalue, psize);
}
ALWAYS_INLINE BOOL binn_list_get_list(void *list, int pos, void **pvalue) {
  return binn_list_get(list, pos, BINN_LIST, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_list_get_map(void *list, int pos, void **pvalue) {
  return binn_list_get(list, pos, BINN_MAP, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_list_get_object(void *list, int pos, void **pvalue) {
  return binn_list_get(list, pos, BINN_OBJECT, pvalue, NULL);
}

/***************************************************************************/

ALWAYS_INLINE BOOL binn_map_get_int8(void *map, int id, signed char *pvalue) {
  return binn_map_get(map, id, BINN_INT8, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_map_get_int16(void *map, int id, short *pvalue) {
  return binn_map_get(map, id, BINN_INT16, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_map_get_int32(void *map, int id, int *pvalue) {
  return binn_map_get(map, id, BINN_INT32, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_map_get_int64(void *map, int id, int64 *pvalue) {
  return binn_map_get(map, id, BINN_INT64, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_map_get_uint8(void *map, int id, unsigned char *pvalue) {
  return binn_map_get(map, id, BINN_UINT8, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_map_get_uint16(void *map, int id, unsigned short *pvalue) {
  return binn_map_get(map, id, BINN_UINT16, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_map_get_uint32(void *map, int id, unsigned int *pvalue) {
  return binn_map_get(map, id, BINN_UINT32, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_map_get_uint64(void *map, int id, uint64 *pvalue) {
  return binn_map_get(map, id, BINN_UINT64, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_map_get_float(void *map, int id, float *pvalue) {
  return binn_map_get(map, id, BINN_FLOAT32, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_map_get_double(void *map, int id, double *pvalue) {
  return binn_map_get(map, id, BINN_FLOAT64, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_map_get_bool(void *map, int id, BOOL *pvalue) {
  return binn_map_get(map, id, BINN_BOOL, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_map_get_str(void *map, int id, char **pvalue) {
  return binn_map_get(map, id, BINN_STRING, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_map_get_blob(void *map, int id, void **pvalue, int *psize) {
  return binn_map_get(map, id, BINN_BLOB, pvalue, psize);
}
ALWAYS_INLINE BOOL binn_map_get_list(void *map, int id, void **pvalue) {
  return binn_map_get(map, id, BINN_LIST, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_map_get_map(void *map, int id, void **pvalue) {
  return binn_map_get(map, id, BINN_MAP, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_map_get_object(void *map, int id, void **pvalue) {
  return binn_map_get(map, id, BINN_OBJECT, pvalue, NULL);
}

/***************************************************************************/

// usage:
//   if (binn_object_get_int32(obj, "key", &value) == FALSE) xxx;

ALWAYS_INLINE BOOL binn_object_get_int8(void *obj, const char *key, signed char *pvalue) {
  return binn_object_get(obj, key, BINN_INT8, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_object_get_int16(void *obj, const char *key, short *pvalue) {
  return binn_object_get(obj, key, BINN_INT16, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_object_get_int32(void *obj, const char *key, int *pvalue) {
  return binn_object_get(obj, key, BINN_INT32, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_object_get_int64(void *obj, const char *key, int64 *pvalue) {
  return binn_object_get(obj, key, BINN_INT64, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_object_get_uint8(void *obj, const char *key, unsigned char *pvalue) {
  return binn_object_get(obj, key, BINN_UINT8, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_object_get_uint16(void *obj, const char *key, unsigned short *pvalue) {
  return binn_object_get(obj, key, BINN_UINT16, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_object_get_uint32(void *obj, const char *key, unsigned int *pvalue) {
  return binn_object_get(obj, key, BINN_UINT32, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_object_get_uint64(void *obj, const char *key, uint64 *pvalue) {
  return binn_object_get(obj, key, BINN_UINT64, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_object_get_float(void *obj, const char *key, float *pvalue) {
  return binn_object_get(obj, key, BINN_FLOAT32, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_object_get_double(void *obj, const char *key, double *pvalue) {
  return binn_object_get(obj, key, BINN_FLOAT64, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_object_get_bool(void *obj, const char *key, BOOL *pvalue) {
  return binn_object_get(obj, key, BINN_BOOL, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_object_get_str(void *obj, const char *key, char **pvalue) {
  return binn_object_get(obj, key, BINN_STRING, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_object_get_blob(void *obj, const char *key, void **pvalue, int *psize) {
  return binn_object_get(obj, key, BINN_BLOB, pvalue, psize);
}
ALWAYS_INLINE BOOL binn_object_get_list(void *obj, const char *key, void **pvalue) {
  return binn_object_get(obj, key, BINN_LIST, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_object_get_map(void *obj, const char *key, void **pvalue) {
  return binn_object_get(obj, key, BINN_MAP, pvalue, NULL);
}
ALWAYS_INLINE BOOL binn_object_get_object(void *obj, const char *key, void **pvalue) {
  return binn_object_get(obj, key, BINN_OBJECT, pvalue, NULL);
}

/***************************************************************************/

BOOL   APIENTRY binn_get_int32(binn *value, int *pint);
BOOL   APIENTRY binn_get_int64(binn *value, int64 *pint);
BOOL   APIENTRY binn_get_double(binn *value, double *pfloat);
BOOL   APIENTRY binn_get_bool(binn *value, BOOL *pbool);
char * APIENTRY binn_get_str(binn *value);

// boolean string values:
// 1, true, yes, on
// 0, false, no, off

// boolean number values:
// !=0 [true]
// ==0 [false]


#ifdef __cplusplus
}
#endif

#endif //BINN_H
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <memory.h>
#include "binn.h"

#define UNUSED(x) (void)(x)
#define roundval(dbl) dbl >= 0.0 ? (int)(dbl + 0.5) : ((dbl - (double)(int)dbl) <= -0.5 ? (int)dbl : (int)(dbl - 0.5))

// magic number:  0x1F 0xb1 0x22 0x1F  =>  0x1FB1221F or 0x1F22B11F
// because the BINN_STORAGE_NOBYTES (binary 000) may not have so many sub-types (BINN_STORAGE_HAS_MORE = 0x10)
#define BINN_MAGIC            0x1F22B11F

#define MAX_BINN_HEADER       9  // [1:type][4:size][4:count]
#define MIN_BINN_SIZE         3  // [1:type][1:size][1:count]
#define CHUNK_SIZE            256  // 1024

#define BINN_STRUCT        1
#define BINN_BUFFER        2

void* (*malloc_fn)(size_t len) = 0;
void* (*realloc_fn)(void *ptr, size_t len) = 0;
void  (*free_fn)(void *ptr) = 0;

/***************************************************************************/

#if defined(__alpha__) || defined(__hppa__) || defined(__mips__) || defined(__powerpc__) || defined(__sparc__)
#define BINN_ONLY_ALIGNED_ACCESS
#elif ( defined(__arm__) || defined(__aarch64__) ) && !defined(__ARM_FEATURE_UNALIGNED)
#define BINN_ONLY_ALIGNED_ACCESS
#endif

#if defined(_WIN32)
#define BIG_ENDIAN      0x1000
#define LITTLE_ENDIAN   0x0001
#define BYTE_ORDER      LITTLE_ENDIAN
#elif defined(__APPLE__)
/* macros already defined */
#elif defined(__OpenBSD__) || defined(__NetBSD__) || defined(__FreeBSD__) || defined(__DragonFly__)
#include <sys/endian.h>
#elif defined(_AIX)
#include <sys/machine.h>
#else
#include <endian.h>
#endif

#ifndef BYTE_ORDER
#error "BYTE_ORDER not defined"
#endif
#ifndef BIG_ENDIAN
#error "BIG_ENDIAN not defined"
#endif
#ifndef LITTLE_ENDIAN
#error "LITTLE_ENDIAN not defined"
#endif
#if BIG_ENDIAN == LITTLE_ENDIAN
#error "BIG_ENDIAN == LITTLE_ENDIAN"
#endif
#if BYTE_ORDER!=BIG_ENDIAN && BYTE_ORDER!=LITTLE_ENDIAN
#error "BYTE_ORDER not supported"
#endif

typedef unsigned short int     u16;
typedef unsigned int           u32;
typedef unsigned long long int u64;

BINN_PRIVATE void copy_be16(u16 *pdest, u16 *psource) {
#if BYTE_ORDER == LITTLE_ENDIAN
  unsigned char *source = (unsigned char *) psource;
  unsigned char *dest = (unsigned char *) pdest;
  dest[0] = source[1];
  dest[1] = source[0];
#else // if BYTE_ORDER == BIG_ENDIAN
#ifdef BINN_ONLY_ALIGNED_ACCESS
  if ((uintptr_t)psource % 2 == 0){  // address aligned to 16 bit
    *pdest = *psource;
  } else {
    unsigned char *source = (unsigned char *) psource;
    unsigned char *dest = (unsigned char *) pdest;
    dest[0] = source[0];  // indexes are the same
    dest[1] = source[1];
  }
#else
  *pdest = *psource;
#endif
#endif
}

BINN_PRIVATE void copy_be32(u32 *pdest, u32 *psource) {
#if BYTE_ORDER == LITTLE_ENDIAN
  unsigned char *source = (unsigned char *) psource;
  unsigned char *dest = (unsigned char *) pdest;
  dest[0] = source[3];
  dest[1] = source[2];
  dest[2] = source[1];
  dest[3] = source[0];
#else // if BYTE_ORDER == BIG_ENDIAN
#ifdef BINN_ONLY_ALIGNED_ACCESS
  if ((uintptr_t)psource % 4 == 0){  // address aligned to 32 bit
    *pdest = *psource;
  } else {
    unsigned char *source = (unsigned char *) psource;
    unsigned char *dest = (unsigned char *) pdest;
    dest[0] = source[0];  // indexes are the same
    dest[1] = source[1];
    dest[2] = source[2];
    dest[3] = source[3];
  }
#else
  *pdest = *psource;
#endif
#endif
}

BINN_PRIVATE void copy_be64(u64 *pdest, u64 *psource) {
#if BYTE_ORDER == LITTLE_ENDIAN
  unsigned char *source = (unsigned char *) psource;
  unsigned char *dest = (unsigned char *) pdest;
  int i;
  for (i=0; i < 8; i++) {
    dest[i] = source[7-i];
  }
#else // if BYTE_ORDER == BIG_ENDIAN
#ifdef BINN_ONLY_ALIGNED_ACCESS
  if ((uintptr_t)psource % 8 == 0){  // address aligned to 64 bit
    *pdest = *psource;
  } else {
    unsigned char *source = (unsigned char *) psource;
    unsigned char *dest = (unsigned char *) pdest;
    int i;
    for (i=0; i < 8; i++) {
      dest[i] = source[i];  // indexes are the same
    }
  }
#else
  *pdest = *psource;
#endif
#endif
}

/***************************************************************************/

#ifndef _MSC_VER
#define stricmp strcasecmp
#define strnicmp strncasecmp
#endif

BINN_PRIVATE BOOL IsValidBinnHeader(void *pbuf, int *ptype, int *pcount, int *psize, int *pheadersize);

/***************************************************************************/

char * APIENTRY binn_version() {
  return BINN_VERSION;
}

/***************************************************************************/

void APIENTRY binn_set_alloc_functions(void* (*new_malloc)(size_t), void* (*new_realloc)(void*,size_t), void (*new_free)(void*)) {

  malloc_fn = new_malloc;
  realloc_fn = new_realloc;
  free_fn = new_free;

}

/***************************************************************************/

BINN_PRIVATE void check_alloc_functions() {

  if (malloc_fn == 0) malloc_fn = &malloc;
  if (realloc_fn == 0) realloc_fn = &realloc;
  if (free_fn == 0) free_fn = &free;

}

/***************************************************************************/

BINN_PRIVATE void * binn_malloc(int size) {
  check_alloc_functions();
  return malloc_fn(size);
}

/***************************************************************************/

BINN_PRIVATE void * binn_memdup(void *src, int size) {
  void *dest;

  if (src == NULL || size <= 0) return NULL;
  dest = binn_malloc(size);
  if (dest == NULL) return NULL;
  memcpy(dest, src, size);
  return dest;

}

/***************************************************************************/

BINN_PRIVATE size_t strlen2(char *str) {

  if (str == NULL) return 0;
  return strlen(str);

}

/***************************************************************************/

int APIENTRY binn_create_type(int storage_type, int data_type_index) {
  if (data_type_index < 0) return -1;
  if ((storage_type < BINN_STORAGE_MIN) || (storage_type > BINN_STORAGE_MAX)) return -1;
  if (data_type_index < 16)
    return storage_type | data_type_index;
  else if (data_type_index < 4096) {
    storage_type |= BINN_STORAGE_HAS_MORE;
    storage_type <<= 8;
    data_type_index >>= 4;
    return storage_type | data_type_index;
  } else
    return -1;
}

/***************************************************************************/

BOOL APIENTRY binn_get_type_info(int long_type, int *pstorage_type, int *pextra_type) {
  int storage_type, extra_type;
  BOOL retval=TRUE;

again:

  if (long_type < 0) {
    goto loc_invalid;
  } else if (long_type <= 0xff) {
    storage_type = long_type & BINN_STORAGE_MASK;
    extra_type = long_type & BINN_TYPE_MASK;
  } else if (long_type <= 0xffff) {
    storage_type = long_type & BINN_STORAGE_MASK16;
    storage_type >>= 8;
    extra_type = long_type & BINN_TYPE_MASK16;
    extra_type >>= 4;
  } else if (long_type & BINN_STORAGE_VIRTUAL) {
    //storage_type = BINN_STORAGE_VIRTUAL;
    //extra_type = xxx;
    long_type &= 0xffff;
    goto again;
  } else {
loc_invalid:
    storage_type = -1;
    extra_type = -1;
    retval = FALSE;
  }

  if (pstorage_type) *pstorage_type = storage_type;
  if (pextra_type) *pextra_type = extra_type;

  return retval;

}

/***************************************************************************/

BOOL APIENTRY binn_create(binn *item, int type, int size, void *pointer) {
  BOOL retval=FALSE;

  switch (type) {
    case BINN_LIST:
    case BINN_MAP:
    case BINN_OBJECT:
      break;
    default:
      goto loc_exit;
  }

  if ((item == NULL) || (size < 0)) goto loc_exit;
  if (size < MIN_BINN_SIZE) {
    if (pointer) goto loc_exit;
    else size = 0;
  }

  memset(item, 0, sizeof(binn));

  if (pointer) {
    item->pre_allocated = TRUE;
    item->pbuf = pointer;
    item->alloc_size = size;
  } else {
    item->pre_allocated = FALSE;
    if (size == 0) size = CHUNK_SIZE;
    pointer = binn_malloc(size);
    if (pointer == 0) return INVALID_BINN;
    item->pbuf = pointer;
    item->alloc_size = size;
  }

  item->header = BINN_MAGIC;
  //item->allocated = FALSE;   -- already zeroed
  item->writable = TRUE;
  item->used_size = MAX_BINN_HEADER;  // save space for the header
  item->type = type;
  //item->count = 0;           -- already zeroed
  item->dirty = TRUE;          // the header is not written to the buffer

  retval = TRUE;

loc_exit:
  return retval;

}

/***************************************************************************/

binn * APIENTRY binn_new(int type, int size, void *pointer) {
  binn *item;

  item = (binn*) binn_malloc(sizeof(binn));

  if (binn_create(item, type, size, pointer) == FALSE) {
    free_fn(item);
    return NULL;
  }

  item->allocated = TRUE;
  return item;

}

/*************************************************************************************/

BOOL APIENTRY binn_create_list(binn *list) {

  return binn_create(list, BINN_LIST, 0, NULL);

}

/*************************************************************************************/

BOOL APIENTRY binn_create_map(binn *map) {

  return binn_create(map, BINN_MAP, 0, NULL);

}

/*************************************************************************************/

BOOL APIENTRY binn_create_object(binn *object) {

  return binn_create(object, BINN_OBJECT, 0, NULL);

}

/***************************************************************************/

binn * APIENTRY binn_list() {
  return binn_new(BINN_LIST, 0, 0);
}

/***************************************************************************/

binn * APIENTRY binn_map() {
  return binn_new(BINN_MAP, 0, 0);
}

/***************************************************************************/

binn * APIENTRY binn_object() {
  return binn_new(BINN_OBJECT, 0, 0);
}

/***************************************************************************/

binn * APIENTRY binn_copy(void *old) {
  int type, count, size, header_size;
  unsigned char *old_ptr = binn_ptr(old);
  binn *item;

  size = 0;
  if (!IsValidBinnHeader(old_ptr, &type, &count, &size, &header_size)) return NULL;

  item = binn_new(type, size - header_size + MAX_BINN_HEADER, NULL);
  if( item ){
    unsigned char *dest;
    dest = ((unsigned char *) item->pbuf) + MAX_BINN_HEADER;
    memcpy(dest, old_ptr + header_size, size - header_size);
    item->used_size = MAX_BINN_HEADER + size - header_size;
    item->count = count;
  }
  return item;

}

/*************************************************************************************/

BOOL APIENTRY binn_load(void *data, binn *value) {

  if ((data == NULL) || (value == NULL)) return FALSE;
  memset(value, 0, sizeof(binn));
  value->header = BINN_MAGIC;
  //value->allocated = FALSE;  --  already zeroed
  //value->writable = FALSE;

  if (binn_is_valid(data, &value->type, &value->count, &value->size) == FALSE) return FALSE;
  value->ptr = data;
  return TRUE;

}

/*************************************************************************************/

binn * APIENTRY binn_open(void *data) {
  binn *item;

  item = (binn*) binn_malloc(sizeof(binn));

  if (binn_load(data, item) == FALSE) {
    free_fn(item);
    return NULL;
  }

  item->allocated = TRUE;
  return item;

}

/***************************************************************************/

BINN_PRIVATE int binn_get_ptr_type(void *ptr) {

  if (ptr == NULL) return 0;

  switch (*(unsigned int *)ptr) {
  case BINN_MAGIC:
    return BINN_STRUCT;
  default:
    return BINN_BUFFER;
  }

}

/***************************************************************************/

BOOL APIENTRY binn_is_struct(void *ptr) {

  if (ptr == NULL) return FALSE;

  if ((*(unsigned int *)ptr) == BINN_MAGIC) {
    return TRUE;
  } else {
    return FALSE;
  }

}

/***************************************************************************/

BINN_PRIVATE int CalcAllocation(int needed_size, int alloc_size) {
  int calc_size;

  calc_size = alloc_size;
  while (calc_size < needed_size) {
    calc_size <<= 1;  // same as *= 2
    //calc_size += CHUNK_SIZE;  -- this is slower than the above line, because there are more reallocations
  }
  return calc_size;

}

/***************************************************************************/

BINN_PRIVATE BOOL CheckAllocation(binn *item, int add_size) {
  int  alloc_size;
  void *ptr;

  if (item->used_size + add_size > item->alloc_size) {
    if (item->pre_allocated) return FALSE;
    alloc_size = CalcAllocation(item->used_size + add_size, item->alloc_size);
    ptr = realloc_fn(item->pbuf, alloc_size);
    if (ptr == NULL) return FALSE;
    item->pbuf = ptr;
    item->alloc_size = alloc_size;
  }

  return TRUE;

}

/***************************************************************************/

#if BYTE_ORDER == BIG_ENDIAN

BINN_PRIVATE int get_storage_size(int storage_type) {

  switch (storage_type) {
  case BINN_STORAGE_NOBYTES:
    return 0;
  case BINN_STORAGE_BYTE:
    return 1;
  case BINN_STORAGE_WORD:
    return 2;
  case BINN_STORAGE_DWORD:
    return 4;
  case BINN_STORAGE_QWORD:
    return 8;
  default:
    return 0;
  }

}

#endif

/***************************************************************************/

BINN_PRIVATE unsigned char * AdvanceDataPos(unsigned char *p, unsigned char *plimit) {
  unsigned char byte;
  int  storage_type, DataSize;

  if (p > plimit) return 0;

  byte = *p; p++;
  storage_type = byte & BINN_STORAGE_MASK;
  if (byte & BINN_STORAGE_HAS_MORE) p++;

  switch (storage_type) {
  case BINN_STORAGE_NOBYTES:
    //p += 0;
    break;
  case BINN_STORAGE_BYTE:
    p ++;
    break;
  case BINN_STORAGE_WORD:
    p += 2;
    break;
  case BINN_STORAGE_DWORD:
    p += 4;
    break;
  case BINN_STORAGE_QWORD:
    p += 8;
    break;
  case BINN_STORAGE_BLOB:
  case BINN_STORAGE_STRING:
    if (p > plimit) return 0;
    DataSize = *((unsigned char*)p);
    if (DataSize & 0x80) {
      if (p + sizeof(int) - 1 > plimit) return 0;
      copy_be32((u32*)&DataSize, (u32*)p);
      DataSize &= 0x7FFFFFFF;
      p+=4;
    } else {
      p++;
    }
    p += DataSize;
    if (storage_type == BINN_STORAGE_STRING) {
      p++;  // null terminator.
    }
    break;
  case BINN_STORAGE_CONTAINER:
    if (p > plimit) return 0;
    DataSize = *((unsigned char*)p);
    if (DataSize & 0x80) {
      if (p + sizeof(int) - 1 > plimit) return 0;
      copy_be32((u32*)&DataSize, (u32*)p);
      DataSize &= 0x7FFFFFFF;
    }
    DataSize--;  // remove the type byte already added before
    p += DataSize;
    break;
  default:
    return 0;
  }

  if (p > plimit) return 0;

  return p;

}

/***************************************************************************/

/*

The id can be stored with 1 to 5 bytes

S = signal bit
X = bit part of id

  0SXX XXXX
  100S XXXX + 1 byte
  101S XXXX + 2 bytes
  110S XXXX + 3 bytes
  1110 0000 + 4 bytes

*/
BINN_PRIVATE int read_map_id(unsigned char **pp, unsigned char *plimit) {
  unsigned char *p, c, sign, type;
  int id, extra_bytes;

  p = *pp;

  c = *p++;

  if (c & 0x80) {
    extra_bytes = ((c & 0x60) >> 5) + 1;
    if (p + extra_bytes > plimit ) {
      *pp = p + extra_bytes;
      return 0;
    }
  }

  type = c & 0xE0;
  sign = c & 0x10;

  if ((c & 0x80) == 0) {
    sign = c & 0x40;
    id = c & 0x3F;
  } else if (type == 0x80) {
    id = c & 0x0F;
    id = (id << 8) | *p++;
  } else if (type == 0xA0) {
    id = c & 0x0F;
    id = (id << 8) | *p++;
    id = (id << 8) | *p++;
  } else if (type == 0xC0) {
    id = c & 0x0F;
    id = (id << 8) | *p++;
    id = (id << 8) | *p++;
    id = (id << 8) | *p++;
  } else if (type == 0xE0) {
    copy_be32((u32*)&id, (u32*)p);
    p += 4;
  } else {
    *pp = plimit + 2;
    return 0;
  }

  if (sign) id = -id;

  *pp = p;

  return id;
}

/***************************************************************************/

BINN_PRIVATE unsigned char * SearchForID(unsigned char *p, int header_size, int size, int numitems, int id) {
  unsigned char *plimit, *base;
  int  i, int32;

  base = p;
  plimit = p + size - 1;
  p += header_size;

  // search for the ID in all the arguments.
  for (i = 0; i < numitems; i++) {
    int32 = read_map_id(&p, plimit);
    if (p > plimit) break;
    // Compare if the IDs are equal.
    if (int32 == id) return p;
    // xxx
    p = AdvanceDataPos(p, plimit);
    if ((p == 0) || (p < base)) break;
  }

  return NULL;

}

/***************************************************************************/

BINN_PRIVATE unsigned char * SearchForKey(unsigned char *p, int header_size, int size, int numitems, const char *key) {
  unsigned char len, *plimit, *base;
  int  i, keylen;

  base = p;
  plimit = p + size - 1;
  p += header_size;

  keylen = strlen(key);

  // search for the key in all the arguments.
  for (i = 0; i < numitems; i++) {
    len = *((unsigned char *)p);
    p++;
    if (p > plimit) break;
    // Compare if the strings are equal.
    if (len > 0) {
      if (strnicmp((char*)p, key, len) == 0) {   // note that there is no null terminator here
        if (keylen == len) {
          p += len;
          return p;
        }
      }
      p += len;
      if (p > plimit) break;
    } else if (len == keylen) {   // in the case of empty string: ""
      return p;
    }
    // xxx
    p = AdvanceDataPos(p, plimit);
    if ((p == 0) || (p < base)) break;
  }

  return NULL;

}

/***************************************************************************/

BINN_PRIVATE BOOL AddValue(binn *item, int type, void *pvalue, int size);

/***************************************************************************/

BINN_PRIVATE BOOL binn_list_add_raw(binn *item, int type, void *pvalue, int size) {

  if ((item == NULL) || (item->type != BINN_LIST) || (item->writable == FALSE)) return FALSE;

  //if (CheckAllocation(item, 4) == FALSE) return FALSE;  // 4 bytes used for data_store and data_format.

  if (AddValue(item, type, pvalue, size) == FALSE) return FALSE;

  item->count++;

  return TRUE;

}

/***************************************************************************/

BINN_PRIVATE BOOL binn_object_set_raw(binn *item, const char *key, int type, void *pvalue, int size) {
  unsigned char *p, len;
  int int32;

  if ((item == NULL) || (item->type != BINN_OBJECT) || (item->writable == FALSE)) return FALSE;

  if (key == NULL) return FALSE;
  int32 = strlen(key);
  if (int32 > 255) return FALSE;

  // is the key already in it?
  p = SearchForKey(item->pbuf, MAX_BINN_HEADER, item->used_size, item->count, key);
  if (p) return FALSE;

  // start adding it

  if (CheckAllocation(item, 1 + int32) == FALSE) return FALSE;  // bytes used for the key size and the key itself.

  p = ((unsigned char *) item->pbuf) + item->used_size;
  len = int32;
  *p = len;
  p++;
  memcpy(p, key, int32);
  int32++;  // now contains the strlen + 1 byte for the len
  item->used_size += int32;

  if (AddValue(item, type, pvalue, size) == FALSE) {
    item->used_size -= int32;
    return FALSE;
  }

  item->count++;

  return TRUE;

}

/***************************************************************************/

BINN_PRIVATE BOOL binn_map_set_raw(binn *item, int id, int type, void *pvalue, int size) {
  unsigned char *base, *p, sign;
  int id_size;

  if ((item == NULL) || (item->type != BINN_MAP) || (item->writable == FALSE)) return FALSE;

  // is the ID already in it?
  p = SearchForID(item->pbuf, MAX_BINN_HEADER, item->used_size, item->count, id);
  if (p) return FALSE;

  // start adding it

  if (CheckAllocation(item, 5) == FALSE) return FALSE;  // max 5 bytes used for the id.

  p = base = ((unsigned char *) item->pbuf) + item->used_size;

  sign = (id < 0);
  if (sign) id = -id;

  if (id <= 0x3F) {
    *p++ = (sign << 6) | id;
  } else if (id <= 0xFFF) {
    *p++ = 0x80 | (sign << 4) | ((id & 0xF00) >> 8);
    *p++ = id & 0xFF;
  } else if (id <= 0xFFFFF) {
    *p++ = 0xA0 | (sign << 4) | ((id & 0xF0000) >> 16);
    *p++ = (id & 0xFF00) >> 8;
    *p++ = id & 0xFF;
  } else if (id <= 0xFFFFFFF) {
    *p++ = 0xC0 | (sign << 4) | ((id & 0xF000000) >> 24);
    *p++ = (id & 0xFF0000) >> 16;
    *p++ = (id & 0xFF00) >> 8;
    *p++ = id & 0xFF;
  } else {
    *p++ = 0xE0;
    if (sign) id = -id;
    copy_be32((u32*)p, (u32*)&id);
    p += 4;
  }

  id_size = (p - base);
  item->used_size += id_size;

  if (AddValue(item, type, pvalue, size) == FALSE) {
    item->used_size -= id_size;
    return FALSE;
  }

  item->count++;

  return TRUE;

}

/***************************************************************************/

BINN_PRIVATE void * compress_int(int *pstorage_type, int *ptype, void *psource) {
  int storage_type, storage_type2, type, type2=0;
  int64  vint = 0;
  uint64 vuint;
  char *pvalue;
#if BYTE_ORDER == BIG_ENDIAN
  int size1, size2;
#endif

  storage_type = *pstorage_type;
  if (storage_type == BINN_STORAGE_BYTE) return psource;

  type = *ptype;

  switch (type) {
  case BINN_INT64:
    vint = *(int64*)psource;
    goto loc_signed;
  case BINN_INT32:
    vint = *(int*)psource;
    goto loc_signed;
  case BINN_INT16:
    vint = *(short*)psource;
    goto loc_signed;
  case BINN_UINT64:
    vuint = *(uint64*)psource;
    goto loc_positive;
  case BINN_UINT32:
    vuint = *(unsigned int*)psource;
    goto loc_positive;
  case BINN_UINT16:
    vuint = *(unsigned short*)psource;
    goto loc_positive;
  }

loc_signed:

  if (vint >= 0) {
    vuint = vint;
    goto loc_positive;
  }

//loc_negative:

  if (vint >= INT8_MIN) {
    type2 = BINN_INT8;
  } else
  if (vint >= INT16_MIN) {
    type2 = BINN_INT16;
  } else
  if (vint >= INT32_MIN) {
    type2 = BINN_INT32;
  }
  goto loc_exit;

loc_positive:

  if (vuint <= UINT8_MAX) {
    type2 = BINN_UINT8;
  } else
  if (vuint <= UINT16_MAX) {
    type2 = BINN_UINT16;
  } else
  if (vuint <= UINT32_MAX) {
    type2 = BINN_UINT32;
  }

loc_exit:

  pvalue = (char *) psource;

  if ((type2) && (type2 != type)) {
    *ptype = type2;
    storage_type2 = binn_get_write_storage(type2);
    *pstorage_type = storage_type2;
#if BYTE_ORDER == BIG_ENDIAN
    size1 = get_storage_size(storage_type);
    size2 = get_storage_size(storage_type2);
    pvalue += (size1 - size2);
#endif
  }

  return pvalue;

}

/***************************************************************************/

BINN_PRIVATE int type_family(int type);

BINN_PRIVATE BOOL AddValue(binn *item, int type, void *pvalue, int size) {
  int int32, ArgSize, storage_type, extra_type;
  unsigned char *p;

  binn_get_type_info(type, &storage_type, &extra_type);

  if (pvalue == NULL) {
    switch (storage_type) {
      case BINN_STORAGE_NOBYTES:
        break;
      case BINN_STORAGE_BLOB:
      case BINN_STORAGE_STRING:
        if (size == 0) break; // the 2 above are allowed to have 0 length
      default:
        return FALSE;
    }
  }

  if ((type_family(type) == BINN_FAMILY_INT) && (item->disable_int_compression == FALSE))
    pvalue = compress_int(&storage_type, &type, pvalue);

  switch (storage_type) {
    case BINN_STORAGE_NOBYTES:
      size = 0;
      ArgSize = size;
      break;
    case BINN_STORAGE_BYTE:
      size = 1;
      ArgSize = size;
      break;
    case BINN_STORAGE_WORD:
      size = 2;
      ArgSize = size;
      break;
    case BINN_STORAGE_DWORD:
      size = 4;
      ArgSize = size;
      break;
    case BINN_STORAGE_QWORD:
      size = 8;
      ArgSize = size;
      break;
    case BINN_STORAGE_BLOB:
      if (size < 0) return FALSE;
      //if (size == 0) ...
      ArgSize = size + 4; // at least this size
      break;
    case BINN_STORAGE_STRING:
      if (size < 0) return FALSE;
      if (size == 0) size = strlen2( (char *) pvalue);
      ArgSize = size + 5; // at least this size
      break;
    case BINN_STORAGE_CONTAINER:
      if (size <= 0) return FALSE;
      ArgSize = size;
      break;
    default:
      return FALSE;
  }

  ArgSize += 2;  // at least 2 bytes used for data_type.
  if (CheckAllocation(item, ArgSize) == FALSE) return FALSE;

  // Gets the pointer to the next place in buffer
  p = ((unsigned char *) item->pbuf) + item->used_size;

  // If the data is not a container, store the data type
  if (storage_type != BINN_STORAGE_CONTAINER) {
    if (type > 255) {
      u16 type16 = type;
      copy_be16((u16*)p, (u16*)&type16);
      p += 2;
      item->used_size += 2;
    } else {
      *p = type;
      p++;
      item->used_size++;
    }
  }

  switch (storage_type) {
    case BINN_STORAGE_NOBYTES:
      // Nothing to do.
      break;
    case BINN_STORAGE_BYTE:
      *((char *) p) = *((char *) pvalue);
      item->used_size += 1;
      break;
    case BINN_STORAGE_WORD:
      copy_be16((u16*)p, (u16*)pvalue);
      item->used_size += 2;
      break;
    case BINN_STORAGE_DWORD:
      copy_be32((u32*)p, (u32*)pvalue);
      item->used_size += 4;
      break;
    case BINN_STORAGE_QWORD:
      copy_be64((u64*)p, (u64*)pvalue);
      item->used_size += 8;
      break;
    case BINN_STORAGE_BLOB:
    case BINN_STORAGE_STRING:
      if (size > 127) {
        int32 = size | 0x80000000;
        copy_be32((u32*)p, (u32*)&int32);
        p += 4;
        item->used_size += 4;
      } else {
        *((unsigned char *) p) = size;
        p++;
        item->used_size++;
      }
      memcpy(p, pvalue, size);
      if (storage_type == BINN_STORAGE_STRING) {
        p += size;
        *((char *) p) = (char) 0;
        size++;  // null terminator
      }
      item->used_size += size;
      break;
    case BINN_STORAGE_CONTAINER:
      memcpy(p, pvalue, size);
      item->used_size += size;
      break;
  }

  item->dirty = TRUE;

  return TRUE;
}

/***************************************************************************/

BINN_PRIVATE BOOL binn_save_header(binn *item) {
  unsigned char byte, *p;
  int int32, size;

  if (item == NULL) return FALSE;

#ifndef BINN_DISABLE_SMALL_HEADER

  p = ((unsigned char *) item->pbuf) + MAX_BINN_HEADER;
  size = item->used_size - MAX_BINN_HEADER + 3;  // at least 3 bytes for the header

  // write the count
  if (item->count > 127) {
    p -= 4;
    size += 3;
    int32 = item->count | 0x80000000;
    copy_be32((u32*)p, (u32*)&int32);
  } else {
    p--;
    *p = (unsigned char) item->count;
  }

  // write the size
  if (size > 127) {
    p -= 4;
    size += 3;
    int32 = size | 0x80000000;
    copy_be32((u32*)p, (u32*)&int32);
  } else {
    p--;
    *p = (unsigned char) size;
  }

  // write the type.
  p--;
  *p = (unsigned char) item->type;

  // set the values
  item->ptr = p;
  item->size = size;

  UNUSED(byte);

#else

  p = (unsigned char *) item->pbuf;

  // write the type.
  byte = item->type;
  *p = byte; p++;
  // write the size
  int32 = item->used_size | 0x80000000;
  copy_be32((u32*)p, (u32*)&int32);
  p+=4;
  // write the count
  int32 = item->count | 0x80000000;
  copy_be32((u32*)p, (u32*)&int32);

  item->ptr = item->pbuf;
  item->size = item->used_size;

#endif

  item->dirty = FALSE;

  return TRUE;

}

/***************************************************************************/

void APIENTRY binn_free(binn *item) {

  if (item == NULL) return;

  if ((item->writable) && (item->pre_allocated == FALSE)) {
    free_fn(item->pbuf);
  }

  if (item->freefn) item->freefn(item->ptr);

  if (item->allocated) {
    free_fn(item);
  } else {
    memset(item, 0, sizeof(binn));
    item->header = BINN_MAGIC;
  }

}

/***************************************************************************/
// free the binn structure but keeps the binn buffer allocated, returning a pointer to it. use the free function to release the buffer later
void * APIENTRY binn_release(binn *item) {
  void *data;

  if (item == NULL) return NULL;

  data = binn_ptr(item);

  if (data > item->pbuf) {
    memmove(item->pbuf, data, item->size);
    data = item->pbuf;
  }

  if (item->allocated) {
    free_fn(item);
  } else {
    memset(item, 0, sizeof(binn));
    item->header = BINN_MAGIC;
  }

  return data;

}

/***************************************************************************/

BINN_PRIVATE BOOL IsValidBinnHeader(void *pbuf, int *ptype, int *pcount, int *psize, int *pheadersize) {
  unsigned char byte, *p, *plimit=0;
  int int32, type, size, count;

  if (pbuf == NULL) return FALSE;

  p = (unsigned char *) pbuf;

  if (psize && *psize > 0) {
    plimit = p + *psize - 1;
  }

  // get the type
  byte = *p; p++;
  if ((byte & BINN_STORAGE_MASK) != BINN_STORAGE_CONTAINER) return FALSE;
  if (byte & BINN_STORAGE_HAS_MORE) return FALSE;
  type = byte;

  switch (type) {
    case BINN_LIST:
    case BINN_MAP:
    case BINN_OBJECT:
      break;
    default:
      return FALSE;
  }

  // get the size
  if (plimit && p > plimit) return FALSE;
  int32 = *((unsigned char*)p);
  if (int32 & 0x80) {
    if (plimit && p + sizeof(int) - 1 > plimit) return FALSE;
    copy_be32((u32*)&int32, (u32*)p);
    int32 &= 0x7FFFFFFF;
    p+=4;
  } else {
    p++;
  }
  size = int32;

  // get the count
  if (plimit && p > plimit) return FALSE;
  int32 = *((unsigned char*)p);
  if (int32 & 0x80) {
    if (plimit && p + sizeof(int) - 1 > plimit) return FALSE;
    copy_be32((u32*)&int32, (u32*)p);
    int32 &= 0x7FFFFFFF;
    p+=4;
  } else {
    p++;
  }
  count = int32;

#if 0
  // get the size
  copy_be32((u32*)&size, (u32*)p);
  size &= 0x7FFFFFFF;
  p+=4;

  // get the count
  copy_be32((u32*)&count, (u32*)p);
  count &= 0x7FFFFFFF;
  p+=4;
#endif

  if ((size < MIN_BINN_SIZE) || (count < 0)) return FALSE;

  // return the values
  if (ptype)  *ptype  = type;
  if (pcount) *pcount = count;
  if (psize && *psize==0) *psize = size;
  if (pheadersize) *pheadersize = (int) (p - (unsigned char*)pbuf);
  return TRUE;
}

/***************************************************************************/

BINN_PRIVATE int binn_buf_type(void *pbuf) {
  int  type;

  if (!IsValidBinnHeader(pbuf, &type, NULL, NULL, NULL)) return INVALID_BINN;

  return type;

}

/***************************************************************************/

BINN_PRIVATE int binn_buf_count(void *pbuf) {
  int  nitems;

  if (!IsValidBinnHeader(pbuf, NULL, &nitems, NULL, NULL)) return 0;

  return nitems;

}

/***************************************************************************/

BINN_PRIVATE int binn_buf_size(void *pbuf) {
  int  size=0;

  if (!IsValidBinnHeader(pbuf, NULL, NULL, &size, NULL)) return 0;

  return size;

}

/***************************************************************************/

void * APIENTRY binn_ptr(void *ptr) {
  binn *item;

  switch (binn_get_ptr_type(ptr)) {
  case BINN_STRUCT:
    item = (binn*) ptr;
    if (item->writable && item->dirty) {
      binn_save_header(item);
    }
    return item->ptr;
  case BINN_BUFFER:
    return ptr;
  default:
    return NULL;
  }

}

/***************************************************************************/

int APIENTRY binn_size(void *ptr) {
  binn *item;

  switch (binn_get_ptr_type(ptr)) {
  case BINN_STRUCT:
    item = (binn*) ptr;
    if (item->writable && item->dirty) {
      binn_save_header(item);
    }
    return item->size;
  case BINN_BUFFER:
    return binn_buf_size(ptr);
  default:
    return 0;
  }

}

/***************************************************************************/

int APIENTRY binn_type(void *ptr) {
  binn *item;

  switch (binn_get_ptr_type(ptr)) {
  case BINN_STRUCT:
    item = (binn*) ptr;
    return item->type;
  case BINN_BUFFER:
    return binn_buf_type(ptr);
  default:
    return -1;
  }

}

/***************************************************************************/

int APIENTRY binn_count(void *ptr) {
  binn *item;

  switch (binn_get_ptr_type(ptr)) {
  case BINN_STRUCT:
    item = (binn*) ptr;
    return item->count;
  case BINN_BUFFER:
    return binn_buf_count(ptr);
  default:
    return -1;
  }

}

/***************************************************************************/

BOOL APIENTRY binn_is_valid_ex(void *ptr, int *ptype, int *pcount, int *psize) {
  int  i, type, count, size, header_size;
  unsigned char *p, *plimit, *base, len;
  void *pbuf;

  pbuf = binn_ptr(ptr);
  if (pbuf == NULL) return FALSE;

  // is there an informed size?
  if (psize && *psize > 0) {
    size = *psize;
  } else {
    size = 0;
  }

  if (!IsValidBinnHeader(pbuf, &type, &count, &size, &header_size)) return FALSE;

  // is there an informed size?
  if (psize && *psize > 0) {
    // is it the same as the one in the buffer?
    if (size != *psize) return FALSE;
  }
  // is there an informed count?
  if (pcount && *pcount > 0) {
    // is it the same as the one in the buffer?
    if (count != *pcount) return FALSE;
  }
  // is there an informed type?
  if (ptype && *ptype != 0) {
    // is it the same as the one in the buffer?
    if (type != *ptype) return FALSE;
  }

  // it could compare the content size with the size informed on the header

  p = (unsigned char *)pbuf;
  base = p;
  plimit = p + size;

  p += header_size;

  // process all the arguments.
  for (i = 0; i < count; i++) {
    switch (type) {
      case BINN_OBJECT:
        // gets the string size (argument name)
        len = *p;
        p++;
        //if (len == 0) goto Invalid;
        // increment the used space
        p += len;
        break;
      case BINN_MAP:
        // increment the used space
        read_map_id(&p, plimit);
        break;
      //case BINN_LIST:
      //  break;
    }
    // xxx
    p = AdvanceDataPos(p, plimit);
    if ((p == 0) || (p < base)) goto Invalid;
  }

  if (ptype  && *ptype==0)  *ptype  = type;
  if (pcount && *pcount==0) *pcount = count;
  if (psize  && *psize==0)  *psize  = size;
  return TRUE;

Invalid:
  return FALSE;

}

/***************************************************************************/

BOOL APIENTRY binn_is_valid(void *ptr, int *ptype, int *pcount, int *psize) {

  if (ptype)  *ptype  = 0;
  if (pcount) *pcount = 0;
  if (psize)  *psize  = 0;

  return binn_is_valid_ex(ptr, ptype, pcount, psize);

}

/***************************************************************************/
/*** INTERNAL FUNCTIONS ****************************************************/
/***************************************************************************/

BINN_PRIVATE BOOL GetValue(unsigned char *p, binn *value) {
  unsigned char byte;
  int   data_type, storage_type;  //, extra_type;
  int   DataSize;
  void *p2;

  if (value == NULL) return FALSE;
  memset(value, 0, sizeof(binn));
  value->header = BINN_MAGIC;
  //value->allocated = FALSE;  --  already zeroed
  //value->writable = FALSE;

  // saves for use with BINN_STORAGE_CONTAINER
  p2 = p;

  // read the data type
  byte = *p; p++;
  storage_type = byte & BINN_STORAGE_MASK;
  if (byte & BINN_STORAGE_HAS_MORE) {
    data_type = byte << 8;
    byte = *p; p++;
    data_type |= byte;
    //extra_type = data_type & BINN_TYPE_MASK16;
  } else {
    data_type = byte;
    //extra_type = byte & BINN_TYPE_MASK;
  }

  //value->storage_type = storage_type;
  value->type = data_type;

  switch (storage_type) {
  case BINN_STORAGE_NOBYTES:
    break;
  case BINN_STORAGE_BYTE:
    value->vuint8 = *((unsigned char *) p);
    value->ptr = p;   //value->ptr = &value->vuint8;
    break;
  case BINN_STORAGE_WORD:
    copy_be16((u16*)&value->vint16, (u16*)p);
    value->ptr = &value->vint16;
    break;
  case BINN_STORAGE_DWORD:
    copy_be32((u32*)&value->vint32, (u32*)p);
    value->ptr = &value->vint32;
    break;
  case BINN_STORAGE_QWORD:
    copy_be64((u64*)&value->vint64, (u64*)p);
    value->ptr = &value->vint64;
    break;
  case BINN_STORAGE_BLOB:
  case BINN_STORAGE_STRING:
    DataSize = *((unsigned char*)p);
    if (DataSize & 0x80) {
      copy_be32((u32*)&DataSize, (u32*)p);
      DataSize &= 0x7FFFFFFF;
      p+=4;
    } else {
      p++;
    }
    value->size = DataSize;
    value->ptr = p;
    break;
  case BINN_STORAGE_CONTAINER:
    value->ptr = p2;  // <-- it returns the pointer to the container, not the data
    if (IsValidBinnHeader(p2, NULL, &value->count, &value->size, NULL) == FALSE) return FALSE;
    break;
  default:
    return FALSE;
  }

  // convert the returned value, if needed

  switch (value->type) {
    case BINN_TRUE:
      value->type = BINN_BOOL;
      value->vbool = TRUE;
      value->ptr = &value->vbool;
      break;
    case BINN_FALSE:
      value->type = BINN_BOOL;
      value->vbool = FALSE;
      value->ptr = &value->vbool;
      break;
#ifdef BINN_EXTENDED
    case BINN_SINGLE_STR:
      value->type = BINN_SINGLE;
      value->vfloat = (float) atof((const char*)value->ptr);  // converts from string to double, and then to float
      value->ptr = &value->vfloat;
      break;
    case BINN_DOUBLE_STR:
      value->type = BINN_DOUBLE;
      value->vdouble = atof((const char*)value->ptr);  // converts from string to double
      value->ptr = &value->vdouble;
      break;
#endif
    /*
    case BINN_DECIMAL:
    case BINN_CURRENCYSTR:
    case BINN_DATE:
    case BINN_DATETIME:
    case BINN_TIME:
    */
  }

  return TRUE;

}

/***************************************************************************/

#if BYTE_ORDER == LITTLE_ENDIAN

// on little-endian devices we store the value so we can return a pointer to integers.
// it's valid only for single-threaded apps. multi-threaded apps must use the _get_ functions instead.

binn local_value;

BINN_PRIVATE void * store_value(binn *value) {

  memcpy(&local_value, value, sizeof(binn));

  switch (binn_get_read_storage(value->type)) {
  case BINN_STORAGE_NOBYTES:
    // return a valid pointer
  case BINN_STORAGE_WORD:
  case BINN_STORAGE_DWORD:
  case BINN_STORAGE_QWORD:
    return &local_value.vint32;  // returns the pointer to the converted value, from big-endian to little-endian
  }

  return value->ptr;   // returns from the on stack value to be thread-safe (for list, map, object, string and blob)

}

#endif

/***************************************************************************/
/*** READ FUNCTIONS ********************************************************/
/***************************************************************************/

BOOL APIENTRY binn_object_get_value(void *ptr, const char *key, binn *value) {
  int type, count, size=0, header_size;
  unsigned char *p;

  ptr = binn_ptr(ptr);
  if ((ptr == 0) || (key == 0) || (value == 0)) return FALSE;

  // check the header
  if (IsValidBinnHeader(ptr, &type, &count, &size, &header_size) == FALSE) return FALSE;

  if (type != BINN_OBJECT) return FALSE;
  if (count == 0) return FALSE;

  p = (unsigned char *) ptr;
  p = SearchForKey(p, header_size, size, count, key);
  if (p == FALSE) return FALSE;

  return GetValue(p, value);

}

/***************************************************************************/

BOOL APIENTRY binn_map_get_value(void* ptr, int id, binn *value) {
  int type, count, size=0, header_size;
  unsigned char *p;

  ptr = binn_ptr(ptr);
  if ((ptr == 0) || (value == 0)) return FALSE;

  // check the header
  if (IsValidBinnHeader(ptr, &type, &count, &size, &header_size) == FALSE) return FALSE;

  if (type != BINN_MAP) return FALSE;
  if (count == 0) return FALSE;

  p = (unsigned char *) ptr;
  p = SearchForID(p, header_size, size, count, id);
  if (p == FALSE) return FALSE;

  return GetValue(p, value);

}

/***************************************************************************/

BOOL APIENTRY binn_list_get_value(void* ptr, int pos, binn *value) {
  int  i, type, count, size=0, header_size;
  unsigned char *p, *plimit, *base;

  ptr = binn_ptr(ptr);
  if ((ptr == 0) || (value == 0)) return FALSE;

  // check the header
  if (IsValidBinnHeader(ptr, &type, &count, &size, &header_size) == FALSE) return FALSE;

  if (type != BINN_LIST) return FALSE;
  if (count == 0) return FALSE;
  if ((pos <= 0) | (pos > count)) return FALSE;
  pos--;  // convert from base 1 to base 0

  p = (unsigned char *) ptr;
  base = p;
  plimit = p + size;
  p += header_size;

  for (i = 0; i < pos; i++) {
    p = AdvanceDataPos(p, plimit);
    if ((p == 0) || (p < base)) return FALSE;
  }

  return GetValue(p, value);

}

/***************************************************************************/
/*** READ PAIR BY POSITION *************************************************/
/***************************************************************************/

BINN_PRIVATE BOOL binn_read_pair(int expected_type, void *ptr, int pos, int *pid, char *pkey, binn *value) {
  int  type, count, size=0, header_size;
  int  i, int32, id = 0, counter=0;
  unsigned char *p, *plimit, *base, *key = NULL, len = 0;

  ptr = binn_ptr(ptr);

  // check the header
  if (IsValidBinnHeader(ptr, &type, &count, &size, &header_size) == FALSE) return FALSE;

  if ((type != expected_type) || (count == 0) || (pos < 1) || (pos > count)) return FALSE;

  p = (unsigned char *) ptr;
  base = p;
  plimit = p + size - 1;
  p += header_size;

  for (i = 0; i < count; i++) {
    switch (type) {
      case BINN_MAP:
        int32 = read_map_id(&p, plimit);
        if (p > plimit) return FALSE;
        id = int32;
        break;
      case BINN_OBJECT:
        len = *((unsigned char *)p); p++;
        if (p > plimit) return FALSE;
        key = p;
        p += len;
        if (p > plimit) return FALSE;
        break;
    }
    counter++;
    if (counter == pos) goto found;
    //
    p = AdvanceDataPos(p, plimit);
    if ((p == 0) || (p < base)) return FALSE;
  }

  return FALSE;

found:

  switch (type) {
    case BINN_MAP:
      if (pid) *pid = id;
      break;
    case BINN_OBJECT:
      if (pkey) {
        memcpy(pkey, key, len);
        pkey[len] = 0;
      }
      break;
  }

  return GetValue(p, value);

}

/***************************************************************************/

BOOL APIENTRY binn_map_get_pair(void *ptr, int pos, int *pid, binn *value) {

  return binn_read_pair(BINN_MAP, ptr, pos, pid, NULL, value);

}

/***************************************************************************/

BOOL APIENTRY binn_object_get_pair(void *ptr, int pos, char *pkey, binn *value) {

  return binn_read_pair(BINN_OBJECT, ptr, pos, NULL, pkey, value);

}

/***************************************************************************/

binn * APIENTRY binn_map_pair(void *map, int pos, int *pid) {
  binn *value;

  value = (binn *) binn_malloc(sizeof(binn));

  if (binn_read_pair(BINN_MAP, map, pos, pid, NULL, value) == FALSE) {
    free_fn(value);
    return NULL;
  }

  value->allocated = TRUE;
  return value;

}

/***************************************************************************/

binn * APIENTRY binn_object_pair(void *obj, int pos, char *pkey) {
  binn *value;

  value = (binn *) binn_malloc(sizeof(binn));

  if (binn_read_pair(BINN_OBJECT, obj, pos, NULL, pkey, value) == FALSE) {
    free_fn(value);
    return NULL;
  }

  value->allocated = TRUE;
  return value;

}

/***************************************************************************/
/***************************************************************************/

void * APIENTRY binn_map_read_pair(void *ptr, int pos, int *pid, int *ptype, int *psize) {
  binn value;

  if (binn_map_get_pair(ptr, pos, pid, &value) == FALSE) return NULL;
  if (ptype) *ptype = value.type;
  if (psize) *psize = value.size;
#if BYTE_ORDER == LITTLE_ENDIAN
  return store_value(&value);
#else
  return value.ptr;
#endif

}

/***************************************************************************/

void * APIENTRY binn_object_read_pair(void *ptr, int pos, char *pkey, int *ptype, int *psize) {
  binn value;

  if (binn_object_get_pair(ptr, pos, pkey, &value) == FALSE) return NULL;
  if (ptype) *ptype = value.type;
  if (psize) *psize = value.size;
#if BYTE_ORDER == LITTLE_ENDIAN
  return store_value(&value);
#else
  return value.ptr;
#endif

}

/***************************************************************************/
/*** SEQUENTIAL READ FUNCTIONS *********************************************/
/***************************************************************************/

BOOL APIENTRY binn_iter_init(binn_iter *iter, void *ptr, int expected_type) {
  int  type, count, size=0, header_size;

  ptr = binn_ptr(ptr);
  if ((ptr == 0) || (iter == 0)) return FALSE;
  memset(iter, 0, sizeof(binn_iter));

  // check the header
  if (IsValidBinnHeader(ptr, &type, &count, &size, &header_size) == FALSE) return FALSE;

  if (type != expected_type) return FALSE;
  //if (count == 0) return FALSE;  -- should not be used

  iter->plimit = (unsigned char *)ptr + size - 1;
  iter->pnext = (unsigned char *)ptr + header_size;
  iter->count = count;
  iter->current = 0;
  iter->type = type;

  return TRUE;
}

/***************************************************************************/

BOOL APIENTRY binn_list_next(binn_iter *iter, binn *value) {
  unsigned char *pnow;

  if ((iter == 0) || (iter->pnext == 0) || (iter->pnext > iter->plimit) || (iter->current > iter->count) || (iter->type != BINN_LIST)) return FALSE;

  iter->current++;
  if (iter->current > iter->count) return FALSE;

  pnow = iter->pnext;
  iter->pnext = AdvanceDataPos(pnow, iter->plimit);
  if (iter->pnext != 0 && iter->pnext < pnow) return FALSE;

  return GetValue(pnow, value);

}

/***************************************************************************/

BINN_PRIVATE BOOL binn_read_next_pair(int expected_type, binn_iter *iter, int *pid, char *pkey, binn *value) {
  int  int32, id;
  unsigned char *p, *key;
  unsigned short len;

  if ((iter == 0) || (iter->pnext == 0) || (iter->pnext > iter->plimit) || (iter->current > iter->count) || (iter->type != expected_type)) return FALSE;

  iter->current++;
  if (iter->current > iter->count) return FALSE;

  p = iter->pnext;

  switch (expected_type) {
    case BINN_MAP:
      int32 = read_map_id(&p, iter->plimit);
      if (p > iter->plimit) return FALSE;
      id = int32;
      if (pid) *pid = id;
      break;
    case BINN_OBJECT:
      len = *((unsigned char *)p); p++;
      key = p;
      p += len;
      if (p > iter->plimit) return FALSE;
      if (pkey) {
        memcpy(pkey, key, len);
        pkey[len] = 0;
      }
      break;
  }

  iter->pnext = AdvanceDataPos(p, iter->plimit);
  if (iter->pnext != 0 && iter->pnext < p) return FALSE;

  return GetValue(p, value);

}

/***************************************************************************/

BOOL APIENTRY binn_map_next(binn_iter *iter, int *pid, binn *value) {

  return binn_read_next_pair(BINN_MAP, iter, pid, NULL, value);

}

/***************************************************************************/

BOOL APIENTRY binn_object_next(binn_iter *iter, char *pkey, binn *value) {

  return binn_read_next_pair(BINN_OBJECT, iter, NULL, pkey, value);

}

/***************************************************************************/
/***************************************************************************/

binn * APIENTRY binn_list_next_value(binn_iter *iter) {
  binn *value;

  value = (binn *) binn_malloc(sizeof(binn));

  if (binn_list_next(iter, value) == FALSE) {
    free_fn(value);
    return NULL;
  }

  value->allocated = TRUE;
  return value;

}

/***************************************************************************/

binn * APIENTRY binn_map_next_value(binn_iter *iter, int *pid) {
  binn *value;

  value = (binn *) binn_malloc(sizeof(binn));

  if (binn_map_next(iter, pid, value) == FALSE) {
    free_fn(value);
    return NULL;
  }

  value->allocated = TRUE;
  return value;

}

/***************************************************************************/

binn * APIENTRY binn_object_next_value(binn_iter *iter, char *pkey) {
  binn *value;

  value = (binn *) binn_malloc(sizeof(binn));

  if (binn_object_next(iter, pkey, value) == FALSE) {
    free_fn(value);
    return NULL;
  }

  value->allocated = TRUE;
  return value;

}

/***************************************************************************/
/***************************************************************************/

void * APIENTRY binn_list_read_next(binn_iter *iter, int *ptype, int *psize) {
  binn value;

  if (binn_list_next(iter, &value) == FALSE) return NULL;
  if (ptype) *ptype = value.type;
  if (psize) *psize = value.size;
#if BYTE_ORDER == LITTLE_ENDIAN
  return store_value(&value);
#else
  return value.ptr;
#endif

}

/***************************************************************************/

void * APIENTRY binn_map_read_next(binn_iter *iter, int *pid, int *ptype, int *psize) {
  binn value;

  if (binn_map_next(iter, pid, &value) == FALSE) return NULL;
  if (ptype) *ptype = value.type;
  if (psize) *psize = value.size;
#if BYTE_ORDER == LITTLE_ENDIAN
  return store_value(&value);
#else
  return value.ptr;
#endif

}

/***************************************************************************/

void * APIENTRY binn_object_read_next(binn_iter *iter, char *pkey, int *ptype, int *psize) {
  binn value;

  if (binn_object_next(iter, pkey, &value) == FALSE) return NULL;
  if (ptype) *ptype = value.type;
  if (psize) *psize = value.size;
#if BYTE_ORDER == LITTLE_ENDIAN
  return store_value(&value);
#else
  return value.ptr;
#endif

}

/*************************************************************************************/
/****** EXTENDED INTERFACE ***********************************************************/
/****** none of the functions above call the functions below *************************/
/*************************************************************************************/

int APIENTRY binn_get_write_storage(int type) {
  int storage_type;

  switch (type) {
    case BINN_SINGLE_STR:
    case BINN_DOUBLE_STR:
      return BINN_STORAGE_STRING;

    case BINN_BOOL:
      return BINN_STORAGE_NOBYTES;

    default:
      binn_get_type_info(type, &storage_type, NULL);
      return storage_type;
  }

}

/*************************************************************************************/

int APIENTRY binn_get_read_storage(int type) {
  int storage_type;

  switch (type) {
#ifdef BINN_EXTENDED
    case BINN_SINGLE_STR:
      return BINN_STORAGE_DWORD;
    case BINN_DOUBLE_STR:
      return BINN_STORAGE_QWORD;
#endif
    case BINN_BOOL:
    case BINN_TRUE:
    case BINN_FALSE:
      return BINN_STORAGE_DWORD;
    default:
      binn_get_type_info(type, &storage_type, NULL);
      return storage_type;
  }

}

/*************************************************************************************/

BINN_PRIVATE BOOL GetWriteConvertedData(int *ptype, void **ppvalue, int *psize) {
  int  type;
  float  f1;
  double d1;
  char pstr[128];

  UNUSED(pstr);
  UNUSED(d1);
  UNUSED(f1);

  type = *ptype;

  if (*ppvalue == NULL) {
    switch (type) {
      case BINN_NULL:
      case BINN_TRUE:
      case BINN_FALSE:
        break;
      case BINN_STRING:
      case BINN_BLOB:
        if (*psize == 0) break;
      default:
        return FALSE;
    }
  }

  switch (type) {
#ifdef BINN_EXTENDED
    case BINN_SINGLE:
      f1 = **(float**)ppvalue;
      d1 = f1;  // convert from float (32bits) to double (64bits)
      type = BINN_SINGLE_STR;
      goto conv_double;
    case BINN_DOUBLE:
      d1 = **(double**)ppvalue;
      type = BINN_DOUBLE_STR;
conv_double:
      // the '%.17e' is more precise than the '%g'
      snprintf(pstr, 127, "%.17e", d1);
      *ppvalue = pstr;
      *ptype = type;
      break;
#endif
    case BINN_DECIMAL:
    case BINN_CURRENCYSTR:
      /*
      if (binn_malloc_extptr(128) == NULL) return FALSE;
      snprintf(sptr, 127, "%E", **ppvalue);
      *ppvalue = sptr;
      */
      return TRUE;  //! temporary
      break;

    case BINN_DATE:
    case BINN_DATETIME:
    case BINN_TIME:
      return TRUE;  //! temporary
      break;

    case BINN_BOOL:
      if (**((BOOL**)ppvalue) == FALSE) {
        type = BINN_FALSE;
      } else {
        type = BINN_TRUE;
      }
      *ptype = type;
      break;

  }

  return TRUE;

}

/*************************************************************************************/

BINN_PRIVATE int type_family(int type)  {

  switch (type) {
    case BINN_LIST:
    case BINN_MAP:
    case BINN_OBJECT:
      return BINN_FAMILY_BINN;

    case BINN_INT8:
    case BINN_INT16:
    case BINN_INT32:
    case BINN_INT64:
    case BINN_UINT8:
    case BINN_UINT16:
    case BINN_UINT32:
    case BINN_UINT64:
      return BINN_FAMILY_INT;

    case BINN_FLOAT32:
    case BINN_FLOAT64:
    //case BINN_SINGLE:
    case BINN_SINGLE_STR:
    //case BINN_DOUBLE:
    case BINN_DOUBLE_STR:
      return BINN_FAMILY_FLOAT;

    case BINN_STRING:
    case BINN_HTML:
    case BINN_CSS:
    case BINN_XML:
    case BINN_JSON:
    case BINN_JAVASCRIPT:
      return BINN_FAMILY_STRING;

    case BINN_BLOB:
    case BINN_JPEG:
    case BINN_GIF:
    case BINN_PNG:
    case BINN_BMP:
      return BINN_FAMILY_BLOB;

    case BINN_DECIMAL:
    case BINN_CURRENCY:
    case BINN_DATE:
    case BINN_TIME:
    case BINN_DATETIME:
      return BINN_FAMILY_STRING;

    case BINN_BOOL:
      return BINN_FAMILY_BOOL;

    case BINN_NULL:
      return BINN_FAMILY_NULL;

    default:
      // if it wasn't found
      return BINN_FAMILY_NONE;
  }

}

/*************************************************************************************/

BINN_PRIVATE int int_type(int type)  {

  switch (type) {
  case BINN_INT8:
  case BINN_INT16:
  case BINN_INT32:
  case BINN_INT64:
    return BINN_SIGNED_INT;

  case BINN_UINT8:
  case BINN_UINT16:
  case BINN_UINT32:
  case BINN_UINT64:
    return BINN_UNSIGNED_INT;

  default:
    return 0;
  }

}

/*************************************************************************************/

BINN_PRIVATE BOOL copy_raw_value(void *psource, void *pdest, int data_store) {

  switch (data_store) {
  case BINN_STORAGE_NOBYTES:
    break;
  case BINN_STORAGE_BYTE:
    *((char *) pdest) = *(char *)psource;
    break;
  case BINN_STORAGE_WORD:
    *((short *) pdest) = *(short *)psource;
    break;
  case BINN_STORAGE_DWORD:
    *((int *) pdest) = *(int *)psource;
    break;
  case BINN_STORAGE_QWORD:
    *((uint64 *) pdest) = *(uint64 *)psource;
    break;
  case BINN_STORAGE_BLOB:
  case BINN_STORAGE_STRING:
  case BINN_STORAGE_CONTAINER:
    *((char **) pdest) = (char *)psource;
    break;
  default:
    return FALSE;
  }

  return TRUE;

}

/*************************************************************************************/

BINN_PRIVATE BOOL copy_int_value(void *psource, void *pdest, int source_type, int dest_type) {
  uint64 vuint64 = 0; int64 vint64 = 0;

  switch (source_type) {
  case BINN_INT8:
    vint64 = *(signed char *)psource;
    break;
  case BINN_INT16:
    vint64 = *(short *)psource;
    break;
  case BINN_INT32:
    vint64 = *(int *)psource;
    break;
  case BINN_INT64:
    vint64 = *(int64 *)psource;
    break;

  case BINN_UINT8:
    vuint64 = *(unsigned char *)psource;
    break;
  case BINN_UINT16:
    vuint64 = *(unsigned short *)psource;
    break;
  case BINN_UINT32:
    vuint64 = *(unsigned int *)psource;
    break;
  case BINN_UINT64:
    vuint64 = *(uint64 *)psource;
    break;

  default:
    return FALSE;
  }


  // copy from int64 to uint64, if possible

  if ((int_type(source_type) == BINN_UNSIGNED_INT) && (int_type(dest_type) == BINN_SIGNED_INT)) {
    if (vuint64 > INT64_MAX) return FALSE;
    vint64 = vuint64;
  } else if ((int_type(source_type) == BINN_SIGNED_INT) && (int_type(dest_type) == BINN_UNSIGNED_INT)) {
    if (vint64 < 0) return FALSE;
    vuint64 = vint64;
  }


  switch (dest_type) {
  case BINN_INT8:
    if ((vint64 < INT8_MIN) || (vint64 > INT8_MAX)) return FALSE;
    *(signed char *)pdest = (signed char) vint64;
    break;
  case BINN_INT16:
    if ((vint64 < INT16_MIN) || (vint64 > INT16_MAX)) return FALSE;
    *(short *)pdest = (short) vint64;
    break;
  case BINN_INT32:
    if ((vint64 < INT32_MIN) || (vint64 > INT32_MAX)) return FALSE;
    *(int *)pdest = (int) vint64;
    break;
  case BINN_INT64:
    *(int64 *)pdest = vint64;
    break;

  case BINN_UINT8:
    if (vuint64 > UINT8_MAX) return FALSE;
    *(unsigned char *)pdest = (unsigned char) vuint64;
    break;
  case BINN_UINT16:
    if (vuint64 > UINT16_MAX) return FALSE;
    *(unsigned short *)pdest = (unsigned short) vuint64;
    break;
  case BINN_UINT32:
    if (vuint64 > UINT32_MAX) return FALSE;
    *(unsigned int *)pdest = (unsigned int) vuint64;
    break;
  case BINN_UINT64:
    *(uint64 *)pdest = vuint64;
    break;

  default:
    return FALSE;
  }

  return TRUE;

}

/*************************************************************************************/

BINN_PRIVATE BOOL copy_float_value(void *psource, void *pdest, int source_type, int dest_type) {

  switch (source_type) {
  case BINN_FLOAT32:
    *(double *)pdest = *(float *)psource;
    break;
  case BINN_FLOAT64:
    *(float *)pdest = (float) *(double *)psource;
    break;
  default:
    return FALSE;
  }

  return TRUE;

}

/*************************************************************************************/

BINN_PRIVATE void zero_value(void *pvalue, int type) {
  //int size=0;

  switch (binn_get_read_storage(type)) {
  case BINN_STORAGE_NOBYTES:
    break;
  case BINN_STORAGE_BYTE:
    *((char *) pvalue) = 0;
    //size=1;
    break;
  case BINN_STORAGE_WORD:
    *((short *) pvalue) = 0;
    //size=2;
    break;
  case BINN_STORAGE_DWORD:
    *((int *) pvalue) = 0;
    //size=4;
    break;
  case BINN_STORAGE_QWORD:
    *((uint64 *) pvalue) = 0;
    //size=8;
    break;
  case BINN_STORAGE_BLOB:
  case BINN_STORAGE_STRING:
  case BINN_STORAGE_CONTAINER:
    *(char **)pvalue = NULL;
    break;
  }

  //if (size>0) memset(pvalue, 0, size);

}

/*************************************************************************************/

BINN_PRIVATE BOOL copy_value(void *psource, void *pdest, int source_type, int dest_type, int data_store) {

  if (type_family(source_type) != type_family(dest_type)) return FALSE;

  if ((type_family(source_type) == BINN_FAMILY_INT) && (source_type != dest_type)) {
    return copy_int_value(psource, pdest, source_type, dest_type);
  } else if ((type_family(source_type) == BINN_FAMILY_FLOAT) && (source_type != dest_type)) {
    return copy_float_value(psource, pdest, source_type, dest_type);
  } else {
    return copy_raw_value(psource, pdest, data_store);
  }

}

/*************************************************************************************/
/*** WRITE FUNCTIONS *****************************************************************/
/*************************************************************************************/

BOOL APIENTRY binn_list_add(binn *list, int type, void *pvalue, int size) {

  if (GetWriteConvertedData(&type, &pvalue, &size) == FALSE) return FALSE;

  return binn_list_add_raw(list, type, pvalue, size);

}

/*************************************************************************************/

BOOL APIENTRY binn_map_set(binn *map, int id, int type, void *pvalue, int size) {

  if (GetWriteConvertedData(&type, &pvalue, &size) == FALSE) return FALSE;

  return binn_map_set_raw(map, id, type, pvalue, size);

}

/*************************************************************************************/

BOOL APIENTRY binn_object_set(binn *obj, const char *key, int type, void *pvalue, int size) {

  if (GetWriteConvertedData(&type, &pvalue, &size) == FALSE) return FALSE;

  return binn_object_set_raw(obj, key, type, pvalue, size);

}

/*************************************************************************************/

// this function is used by the wrappers
BOOL APIENTRY binn_add_value(binn *item, int binn_type, int id, char *name, int type, void *pvalue, int size) {

  switch (binn_type) {
    case BINN_LIST:
      return binn_list_add(item, type, pvalue, size);
    case BINN_MAP:
      return binn_map_set(item, id, type, pvalue, size);
    case BINN_OBJECT:
      return binn_object_set(item, name, type, pvalue, size);
    default:
      return FALSE;
  }

}

/*************************************************************************************/
/*************************************************************************************/

BOOL APIENTRY binn_list_add_new(binn *list, binn *value) {
  BOOL retval;

  retval = binn_list_add_value(list, value);
  if (value) free_fn(value);
  return retval;

}

/*************************************************************************************/

BOOL APIENTRY binn_map_set_new(binn *map, int id, binn *value) {
  BOOL retval;

  retval = binn_map_set_value(map, id, value);
  if (value) free_fn(value);
  return retval;

}

/*************************************************************************************/

BOOL APIENTRY binn_object_set_new(binn *obj, const char *key, binn *value) {
  BOOL retval;

  retval = binn_object_set_value(obj, key, value);
  if (value) free_fn(value);
  return retval;

}

/*************************************************************************************/
/*** READ FUNCTIONS ******************************************************************/
/*************************************************************************************/

binn * APIENTRY binn_list_value(void *ptr, int pos) {
  binn *value;

  value = (binn *) binn_malloc(sizeof(binn));

  if (binn_list_get_value(ptr, pos, value) == FALSE) {
    free_fn(value);
    return NULL;
  }

  value->allocated = TRUE;
  return value;

}

/*************************************************************************************/

binn * APIENTRY binn_map_value(void *ptr, int id) {
  binn *value;

  value = (binn *) binn_malloc(sizeof(binn));

  if (binn_map_get_value(ptr, id, value) == FALSE) {
    free_fn(value);
    return NULL;
  }

  value->allocated = TRUE;
  return value;

}

/*************************************************************************************/

binn * APIENTRY binn_object_value(void *ptr, const char *key) {
  binn *value;

  value = (binn *) binn_malloc(sizeof(binn));

  if (binn_object_get_value(ptr, key, value) == FALSE) {
    free_fn(value);
    return NULL;
  }

  value->allocated = TRUE;
  return value;

}

/***************************************************************************/
/***************************************************************************/

void * APIENTRY binn_list_read(void *list, int pos, int *ptype, int *psize) {
  binn value;

  if (binn_list_get_value(list, pos, &value) == FALSE) return NULL;
  if (ptype) *ptype = value.type;
  if (psize) *psize = value.size;
#if BYTE_ORDER == LITTLE_ENDIAN
  return store_value(&value);
#else
  return value.ptr;
#endif

}

/***************************************************************************/

void * APIENTRY binn_map_read(void *map, int id, int *ptype, int *psize) {
  binn value;

  if (binn_map_get_value(map, id, &value) == FALSE) return NULL;
  if (ptype) *ptype = value.type;
  if (psize) *psize = value.size;
#if BYTE_ORDER == LITTLE_ENDIAN
  return store_value(&value);
#else
  return value.ptr;
#endif

}

/***************************************************************************/

void * APIENTRY binn_object_read(void *obj, const char *key, int *ptype, int *psize) {
  binn value;

  if (binn_object_get_value(obj, key, &value) == FALSE) return NULL;
  if (ptype) *ptype = value.type;
  if (psize) *psize = value.size;
#if BYTE_ORDER == LITTLE_ENDIAN
  return store_value(&value);
#else
  return value.ptr;
#endif

}

/***************************************************************************/
/***************************************************************************/

BOOL APIENTRY binn_list_get(void *ptr, int pos, int type, void *pvalue, int *psize) {
  binn value;
  int storage_type;

  storage_type = binn_get_read_storage(type);
  if ((storage_type != BINN_STORAGE_NOBYTES) && (pvalue == NULL)) return FALSE;

  zero_value(pvalue, type);

  if (binn_list_get_value(ptr, pos, &value) == FALSE) return FALSE;

  if (copy_value(value.ptr, pvalue, value.type, type, storage_type) == FALSE) return FALSE;

  if (psize) *psize = value.size;

  return TRUE;

}

/***************************************************************************/

BOOL APIENTRY binn_map_get(void *ptr, int id, int type, void *pvalue, int *psize) {
  binn value;
  int storage_type;

  storage_type = binn_get_read_storage(type);
  if ((storage_type != BINN_STORAGE_NOBYTES) && (pvalue == NULL)) return FALSE;

  zero_value(pvalue, type);

  if (binn_map_get_value(ptr, id, &value) == FALSE) return FALSE;

  if (copy_value(value.ptr, pvalue, value.type, type, storage_type) == FALSE) return FALSE;

  if (psize) *psize = value.size;

  return TRUE;

}

/***************************************************************************/

//   if (binn_object_get(obj, "multiplier", BINN_INT32, &multiplier, NULL) == FALSE) xxx;

BOOL APIENTRY binn_object_get(void *ptr, const char *key, int type, void *pvalue, int *psize) {
  binn value;
  int storage_type;

  storage_type = binn_get_read_storage(type);
  if ((storage_type != BINN_STORAGE_NOBYTES) && (pvalue == NULL)) return FALSE;

  zero_value(pvalue, type);

  if (binn_object_get_value(ptr, key, &value) == FALSE) return FALSE;

  if (copy_value(value.ptr, pvalue, value.type, type, storage_type) == FALSE) return FALSE;

  if (psize) *psize = value.size;

  return TRUE;

}

/***************************************************************************/
/***************************************************************************/

// these functions below may not be implemented as inline functions, because
// they use a lot of space, even for the variable. so they will be exported.

// but what about using as static?
//    is there any problem with wrappers? can these wrappers implement these functions using the header?
//    if as static, will they be present even on modules that don't use the functions?

signed char APIENTRY binn_list_int8(void *list, int pos) {
  signed char value;

  binn_list_get(list, pos, BINN_INT8, &value, NULL);

  return value;
}

short APIENTRY binn_list_int16(void *list, int pos) {
  short value;

  binn_list_get(list, pos, BINN_INT16, &value, NULL);

  return value;
}

int APIENTRY binn_list_int32(void *list, int pos) {
  int value;

  binn_list_get(list, pos, BINN_INT32, &value, NULL);

  return value;
}

int64 APIENTRY binn_list_int64(void *list, int pos) {
  int64 value;

  binn_list_get(list, pos, BINN_INT64, &value, NULL);

  return value;
}

unsigned char APIENTRY binn_list_uint8(void *list, int pos) {
  unsigned char value;

  binn_list_get(list, pos, BINN_UINT8, &value, NULL);

  return value;
}

unsigned short APIENTRY binn_list_uint16(void *list, int pos) {
  unsigned short value;

  binn_list_get(list, pos, BINN_UINT16, &value, NULL);

  return value;
}

unsigned int APIENTRY binn_list_uint32(void *list, int pos) {
  unsigned int value;

  binn_list_get(list, pos, BINN_UINT32, &value, NULL);

  return value;
}

uint64 APIENTRY binn_list_uint64(void *list, int pos) {
  uint64 value;

  binn_list_get(list, pos, BINN_UINT64, &value, NULL);

  return value;
}

float APIENTRY binn_list_float(void *list, int pos) {
  float value;

  binn_list_get(list, pos, BINN_FLOAT32, &value, NULL);

  return value;
}

double APIENTRY binn_list_double(void *list, int pos) {
  double value;

  binn_list_get(list, pos, BINN_FLOAT64, &value, NULL);

  return value;
}

BOOL APIENTRY binn_list_bool(void *list, int pos) {
  BOOL value;

  binn_list_get(list, pos, BINN_BOOL, &value, NULL);

  return value;
}

BOOL APIENTRY binn_list_null(void *list, int pos) {

  return binn_list_get(list, pos, BINN_NULL, NULL, NULL);

}

char * APIENTRY binn_list_str(void *list, int pos) {
  char *value;

  binn_list_get(list, pos, BINN_STRING, &value, NULL);

  return value;
}

void * APIENTRY binn_list_blob(void *list, int pos, int *psize) {
  void *value;

  binn_list_get(list, pos, BINN_BLOB, &value, psize);

  return value;
}

void * APIENTRY binn_list_list(void *list, int pos) {
  void *value;

  binn_list_get(list, pos, BINN_LIST, &value, NULL);

  return value;
}

void * APIENTRY binn_list_map(void *list, int pos) {
  void *value;

  binn_list_get(list, pos, BINN_MAP, &value, NULL);

  return value;
}

void * APIENTRY binn_list_object(void *list, int pos) {
  void *value;

  binn_list_get(list, pos, BINN_OBJECT, &value, NULL);

  return value;
}

/***************************************************************************/

signed char APIENTRY binn_map_int8(void *map, int id) {
  signed char value;

  binn_map_get(map, id, BINN_INT8, &value, NULL);

  return value;
}

short APIENTRY binn_map_int16(void *map, int id) {
  short value;

  binn_map_get(map, id, BINN_INT16, &value, NULL);

  return value;
}

int APIENTRY binn_map_int32(void *map, int id) {
  int value;

  binn_map_get(map, id, BINN_INT32, &value, NULL);

  return value;
}

int64 APIENTRY binn_map_int64(void *map, int id) {
  int64 value;

  binn_map_get(map, id, BINN_INT64, &value, NULL);

  return value;
}

unsigned char APIENTRY binn_map_uint8(void *map, int id) {
  unsigned char value;

  binn_map_get(map, id, BINN_UINT8, &value, NULL);

  return value;
}

unsigned short APIENTRY binn_map_uint16(void *map, int id) {
  unsigned short value;

  binn_map_get(map, id, BINN_UINT16, &value, NULL);

  return value;
}

unsigned int APIENTRY binn_map_uint32(void *map, int id) {
  unsigned int value;

  binn_map_get(map, id, BINN_UINT32, &value, NULL);

  return value;
}

uint64 APIENTRY binn_map_uint64(void *map, int id) {
  uint64 value;

  binn_map_get(map, id, BINN_UINT64, &value, NULL);

  return value;
}

float APIENTRY binn_map_float(void *map, int id) {
  float value;

  binn_map_get(map, id, BINN_FLOAT32, &value, NULL);

  return value;
}

double APIENTRY binn_map_double(void *map, int id) {
  double value;

  binn_map_get(map, id, BINN_FLOAT64, &value, NULL);

  return value;
}

BOOL APIENTRY binn_map_bool(void *map, int id) {
  BOOL value;

  binn_map_get(map, id, BINN_BOOL, &value, NULL);

  return value;
}

BOOL APIENTRY binn_map_null(void *map, int id) {

  return binn_map_get(map, id, BINN_NULL, NULL, NULL);

}

char * APIENTRY binn_map_str(void *map, int id) {
  char *value;

  binn_map_get(map, id, BINN_STRING, &value, NULL);

  return value;
}

void * APIENTRY binn_map_blob(void *map, int id, int *psize) {
  void *value;

  binn_map_get(map, id, BINN_BLOB, &value, psize);

  return value;
}

void * APIENTRY binn_map_list(void *map, int id) {
  void *value;

  binn_map_get(map, id, BINN_LIST, &value, NULL);

  return value;
}

void * APIENTRY binn_map_map(void *map, int id) {
  void *value;

  binn_map_get(map, id, BINN_MAP, &value, NULL);

  return value;
}

void * APIENTRY binn_map_object(void *map, int id) {
  void *value;

  binn_map_get(map, id, BINN_OBJECT, &value, NULL);

  return value;
}

/***************************************************************************/

signed char APIENTRY binn_object_int8(void *obj, const char *key) {
  signed char value;

  binn_object_get(obj, key, BINN_INT8, &value, NULL);

  return value;
}

short APIENTRY binn_object_int16(void *obj, const char *key) {
  short value;

  binn_object_get(obj, key, BINN_INT16, &value, NULL);

  return value;
}

int APIENTRY binn_object_int32(void *obj, const char *key) {
  int value;

  binn_object_get(obj, key, BINN_INT32, &value, NULL);

  return value;
}

int64 APIENTRY binn_object_int64(void *obj, const char *key) {
  int64 value;

  binn_object_get(obj, key, BINN_INT64, &value, NULL);

  return value;
}

unsigned char APIENTRY binn_object_uint8(void *obj, const char *key) {
  unsigned char value;

  binn_object_get(obj, key, BINN_UINT8, &value, NULL);

  return value;
}

unsigned short APIENTRY binn_object_uint16(void *obj, const char *key) {
  unsigned short value;

  binn_object_get(obj, key, BINN_UINT16, &value, NULL);

  return value;
}

unsigned int APIENTRY binn_object_uint32(void *obj, const char *key) {
  unsigned int value;

  binn_object_get(obj, key, BINN_UINT32, &value, NULL);

  return value;
}

uint64 APIENTRY binn_object_uint64(void *obj, const char *key) {
  uint64 value;

  binn_object_get(obj, key, BINN_UINT64, &value, NULL);

  return value;
}

float APIENTRY binn_object_float(void *obj, const char *key) {
  float value;

  binn_object_get(obj, key, BINN_FLOAT32, &value, NULL);

  return value;
}

double APIENTRY binn_object_double(void *obj, const char *key) {
  double value;

  binn_object_get(obj, key, BINN_FLOAT64, &value, NULL);

  return value;
}

BOOL APIENTRY binn_object_bool(void *obj, const char *key) {
  BOOL value;

  binn_object_get(obj, key, BINN_BOOL, &value, NULL);

  return value;
}

BOOL APIENTRY binn_object_null(void *obj, const char *key) {

  return binn_object_get(obj, key, BINN_NULL, NULL, NULL);

}

char * APIENTRY binn_object_str(void *obj, const char *key) {
  char *value;

  binn_object_get(obj, key, BINN_STRING, &value, NULL);

  return value;
}

void * APIENTRY binn_object_blob(void *obj, const char *key, int *psize) {
  void *value;

  binn_object_get(obj, key, BINN_BLOB, &value, psize);

  return value;
}

void * APIENTRY binn_object_list(void *obj, const char *key) {
  void *value;

  binn_object_get(obj, key, BINN_LIST, &value, NULL);

  return value;
}

void * APIENTRY binn_object_map(void *obj, const char *key) {
  void *value;

  binn_object_get(obj, key, BINN_MAP, &value, NULL);

  return value;
}

void * APIENTRY binn_object_object(void *obj, const char *key) {
  void *value;

  binn_object_get(obj, key, BINN_OBJECT, &value, NULL);

  return value;
}

/*************************************************************************************/
/*************************************************************************************/

BINN_PRIVATE binn * binn_alloc_item() {
  binn *item;
  item = (binn *) binn_malloc(sizeof(binn));
  if (item) {
    memset(item, 0, sizeof(binn));
    item->header = BINN_MAGIC;
    item->allocated = TRUE;
    //item->writable = FALSE;  -- already zeroed
  }
  return item;
}

/*************************************************************************************/

binn * APIENTRY binn_value(int type, void *pvalue, int size, binn_mem_free freefn) {
  int storage_type;
  binn *item = binn_alloc_item();
  if (item) {
    item->type = type;
    binn_get_type_info(type, &storage_type, NULL);
    switch (storage_type) {
    case BINN_STORAGE_NOBYTES:
      break;
    case BINN_STORAGE_STRING:
      if (size == 0) size = strlen((char*)pvalue) + 1;
    case BINN_STORAGE_BLOB:
    case BINN_STORAGE_CONTAINER:
      if (freefn == BINN_TRANSIENT) {
        item->ptr = binn_memdup(pvalue, size);
        if (item->ptr == NULL) {
          free_fn(item);
          return NULL;
        }
        item->freefn = free_fn;
        if (storage_type == BINN_STORAGE_STRING) size--;
      } else {
        item->ptr = pvalue;
        item->freefn = freefn;
      }
      item->size = size;
      break;
    default:
      item->ptr = &item->vint32;
      copy_raw_value(pvalue, item->ptr, storage_type);
    }
  }
  return item;
}

/*************************************************************************************/

BOOL APIENTRY binn_set_string(binn *item, char *str, binn_mem_free pfree) {

  if (item == NULL || str == NULL) return FALSE;

  if (pfree == BINN_TRANSIENT) {
    item->ptr = binn_memdup(str, strlen(str) + 1);
    if (item->ptr == NULL) return FALSE;
    item->freefn = free_fn;
  } else {
    item->ptr = str;
    item->freefn = pfree;
  }

  item->type = BINN_STRING;
  return TRUE;

}

/*************************************************************************************/

BOOL APIENTRY binn_set_blob(binn *item, void *ptr, int size, binn_mem_free pfree) {

  if (item == NULL || ptr == NULL) return FALSE;

  if (pfree == BINN_TRANSIENT) {
    item->ptr = binn_memdup(ptr, size);
    if (item->ptr == NULL) return FALSE;
    item->freefn = free_fn;
  } else {
    item->ptr = ptr;
    item->freefn = pfree;
  }

  item->type = BINN_BLOB;
  item->size = size;
  return TRUE;

}

/*************************************************************************************/
/*** READ CONVERTED VALUE ************************************************************/
/*************************************************************************************/

#ifdef _MSC_VER
#define atoi64 _atoi64
#else
int64 atoi64(char *str) {
  int64 retval;
  int is_negative=0;

  if (*str == '-') {
    is_negative = 1;
    str++;
  }
  retval = 0;
  for (; *str; str++) {
    retval = 10 * retval + (*str - '0');
  }
  if (is_negative) retval *= -1;
  return retval;
}
#endif

/*****************************************************************************/

BINN_PRIVATE BOOL is_integer(char *p) {
  BOOL retval;

  if (p == NULL) return FALSE;
  if (*p == '-') p++;
  if (*p == 0) return FALSE;

  retval = TRUE;

  for (; *p; p++) {
    if ( (*p < '0') || (*p > '9') ) {
      retval = FALSE;
    }
  }

  return retval;
}

/*****************************************************************************/

BINN_PRIVATE BOOL is_float(char *p) {
  BOOL retval, number_found=FALSE;

  if (p == NULL) return FALSE;
  if (*p == '-') p++;
  if (*p == 0) return FALSE;

  retval = TRUE;

  for (; *p; p++) {
    if ((*p == '.') || (*p == ',')) {
      if (!number_found) retval = FALSE;
    } else if ( (*p >= '0') && (*p <= '9') ) {
      number_found = TRUE;
    } else {
      return FALSE;
    }
  }

  return retval;
}

/*************************************************************************************/

BINN_PRIVATE BOOL is_bool_str(char *str, BOOL *pbool) {
  int64  vint;
  double vdouble;

  if (str == NULL || pbool == NULL) return FALSE;

  if (stricmp(str, "true") == 0) goto loc_true;
  if (stricmp(str, "yes") == 0) goto loc_true;
  if (stricmp(str, "on") == 0) goto loc_true;
  //if (stricmp(str, "1") == 0) goto loc_true;

  if (stricmp(str, "false") == 0) goto loc_false;
  if (stricmp(str, "no") == 0) goto loc_false;
  if (stricmp(str, "off") == 0) goto loc_false;
  //if (stricmp(str, "0") == 0) goto loc_false;

  if (is_integer(str)) {
    vint = atoi64(str);
    *pbool = (vint != 0) ? TRUE : FALSE;
    return TRUE;
  } else if (is_float(str)) {
    vdouble = atof(str);
    *pbool = (vdouble != 0) ? TRUE : FALSE;
    return TRUE;
  }

  return FALSE;

loc_true:
  *pbool = TRUE;
  return TRUE;

loc_false:
  *pbool = FALSE;
  return TRUE;

}

/*************************************************************************************/

BOOL APIENTRY binn_get_int32(binn *value, int *pint) {

  if (value == NULL || pint == NULL) return FALSE;

  if (type_family(value->type) == BINN_FAMILY_INT) {
    return copy_int_value(value->ptr, pint, value->type, BINN_INT32);
  }

  switch (value->type) {
  case BINN_FLOAT:
    if ((value->vfloat < INT32_MIN) || (value->vfloat > INT32_MAX)) return FALSE;
    *pint = roundval(value->vfloat);
    break;
  case BINN_DOUBLE:
    if ((value->vdouble < INT32_MIN) || (value->vdouble > INT32_MAX)) return FALSE;
    *pint = roundval(value->vdouble);
    break;
  case BINN_STRING:
    if (is_integer((char*)value->ptr))
      *pint = atoi((char*)value->ptr);
    else if (is_float((char*)value->ptr))
      *pint = roundval(atof((char*)value->ptr));
    else
      return FALSE;
    break;
  case BINN_BOOL:
    *pint = value->vbool;
    break;
  default:
    return FALSE;
  }

  return TRUE;
}

/*************************************************************************************/

BOOL APIENTRY binn_get_int64(binn *value, int64 *pint) {

  if (value == NULL || pint == NULL) return FALSE;

  if (type_family(value->type) == BINN_FAMILY_INT) {
    return copy_int_value(value->ptr, pint, value->type, BINN_INT64);
  }

  switch (value->type) {
  case BINN_FLOAT:
    if ((value->vfloat < INT64_MIN) || (value->vfloat > INT64_MAX)) return FALSE;
    *pint = roundval(value->vfloat);
    break;
  case BINN_DOUBLE:
    if ((value->vdouble < INT64_MIN) || (value->vdouble > INT64_MAX)) return FALSE;
    *pint = roundval(value->vdouble);
    break;
  case BINN_STRING:
    if (is_integer((char*)value->ptr))
      *pint = atoi64((char*)value->ptr);
    else if (is_float((char*)value->ptr))
      *pint = roundval(atof((char*)value->ptr));
    else
      return FALSE;
    break;
  case BINN_BOOL:
    *pint = value->vbool;
    break;
  default:
    return FALSE;
  }

  return TRUE;
}

/*************************************************************************************/

BOOL APIENTRY binn_get_double(binn *value, double *pfloat) {
  int64 vint;

  if (value == NULL || pfloat == NULL) return FALSE;

  if (type_family(value->type) == BINN_FAMILY_INT) {
    if (copy_int_value(value->ptr, &vint, value->type, BINN_INT64) == FALSE) return FALSE;
    *pfloat = (double) vint;
    return TRUE;
  }

  switch (value->type) {
  case BINN_FLOAT:
    *pfloat = value->vfloat;
    break;
  case BINN_DOUBLE:
    *pfloat = value->vdouble;
    break;
  case BINN_STRING:
    if (is_integer((char*)value->ptr))
      *pfloat = (double) atoi64((char*)value->ptr);
    else if (is_float((char*)value->ptr))
      *pfloat = atof((char*)value->ptr);
    else
      return FALSE;
    break;
  case BINN_BOOL:
    *pfloat = value->vbool;
    break;
  default:
    return FALSE;
  }

  return TRUE;
}

/*************************************************************************************/

BOOL APIENTRY binn_get_bool(binn *value, BOOL *pbool) {
  int64 vint;

  if (value == NULL || pbool == NULL) return FALSE;

  if (type_family(value->type) == BINN_FAMILY_INT) {
    if (copy_int_value(value->ptr, &vint, value->type, BINN_INT64) == FALSE) return FALSE;
    *pbool = (vint != 0) ? TRUE : FALSE;
    return TRUE;
  }

  switch (value->type) {
  case BINN_BOOL:
    *pbool = value->vbool;
    break;
  case BINN_FLOAT:
    *pbool = (value->vfloat != 0) ? TRUE : FALSE;
    break;
  case BINN_DOUBLE:
    *pbool = (value->vdouble != 0) ? TRUE : FALSE;
    break;
  case BINN_STRING:
    return is_bool_str((char*)value->ptr, pbool);
  default:
    return FALSE;
  }

  return TRUE;
}

/*************************************************************************************/

char * APIENTRY binn_get_str(binn *value) {
  int64 vint;
  char buf[128];

  if (value == NULL) return NULL;

  if (type_family(value->type) == BINN_FAMILY_INT) {
    if (copy_int_value(value->ptr, &vint, value->type, BINN_INT64) == FALSE) return NULL;
    snprintf(buf, sizeof buf, "%" INT64_FORMAT, vint);
    goto loc_convert_value;
  }

  switch (value->type) {
  case BINN_FLOAT:
    value->vdouble = value->vfloat;
  case BINN_DOUBLE:
    snprintf(buf, sizeof buf, "%g", value->vdouble);
    goto loc_convert_value;
  case BINN_STRING:
    return (char*) value->ptr;
  case BINN_BOOL:
    if (value->vbool)
      strcpy(buf, "true");
    else
      strcpy(buf, "false");
    goto loc_convert_value;
  }

  return NULL;

loc_convert_value:

  //value->vint64 = 0;
  value->ptr = strdup(buf);
  if (value->ptr == NULL) return NULL;
  value->freefn = free;
  value->type = BINN_STRING;
  return (char*) value->ptr;

}

/*************************************************************************************/
/*** GENERAL FUNCTIONS ***************************************************************/
/*************************************************************************************/

BOOL APIENTRY binn_is_container(binn *item) {

  if (item == NULL) return FALSE;

  switch (item->type) {
  case BINN_LIST:
  case BINN_MAP:
  case BINN_OBJECT:
    return TRUE;
  default:
    return FALSE;
  }

}

/*************************************************************************************/
   """
    print(tokenizer.text_token_count(code))