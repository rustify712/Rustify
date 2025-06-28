// Rust实现单向链表
// 从C版本转换而来

pub type SListValue = i32;

pub struct SListEntry {
    pub data: SListValue,
    pub next: Option<Box<SListEntry>>,
}

impl SListEntry {
    // 释放整个链表
    pub fn free(list: &mut Option<Box<SListEntry>>) {
        while let Some(mut entry) = list.take() {
            *list = entry.next.take();
        }
    }

    // 在链表头部插入新节点
    pub fn prepend(list: &mut Option<Box<SListEntry>>, data: SListValue) -> &mut Option<Box<SListEntry>> {
        let new_entry = Box::new(SListEntry {
            data,
            next: list.take(),
        });
        *list = Some(new_entry);
        list
    }

    // 在链表尾部插入新节点
    pub fn append(list: &mut Option<Box<SListEntry>>, data: SListValue) -> &mut Option<Box<SListEntry>> {
        let new_entry = Box::new(SListEntry {
            data,
            next: None,
        });

        if list.is_none() {
            *list = Some(new_entry);
            return list;
        }

        let mut current = list;
        while let Some(entry) = current {
            if entry.next.is_none() {
                entry.next = Some(new_entry);
                break;
            }
            current = &mut entry.next;
        }

        list
    }
}