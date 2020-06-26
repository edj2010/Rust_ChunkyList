///
/// Rust Chunky_List Implementation
/// By Evan Johnson
/// Jun 21 2020
/// 

const CHUNK_SIZE: usize = 8;
const SPLIT_POINT: usize = CHUNK_SIZE/2;

/*
use std::collections::LinkedList;
use std::collections::linked_list::Iter as LinkedListIter;
use std::collections::linked_list::IterMut as LinkedListIterMut;
use std::collections::linked_list::IntoIter as LinkedListIntoIter;
use std::collections::linked_list::Cursor as LinkedListCursor;
use std::collections::linked_list::CursorMut as LinkedListCursorMut;
*/

extern crate linked_deque;
use linked_deque::LinkedDeque as LinkedList;
use linked_deque::Iter as LinkedListIter;
use linked_deque::IterMut as LinkedListIterMut;
use linked_deque::IntoIter as LinkedListIntoIter;
use linked_deque::Cursor as LinkedListCursor;
use linked_deque::CursorMut as LinkedListCursorMut;

use std::slice::Iter as ArrIter;
use std::slice::IterMut as ArrIterMut;

use std::iter::FromIterator;
use std::fmt;

use std::cmp;
use std::cmp::Ordering;
use std::ops::AddAssign;

/// A chunk class to handle our individual chunks
#[derive(Default, Clone)]
struct Chunk<T: Default + Copy> {
    /// A chunk stores data in a small, fixed array
    /// Important: chunks should never be empty
    data: [T; CHUNK_SIZE],
    len: usize
}

impl<T: Default + Copy> Chunk<T>{
    fn new() -> Self { Default::default() }

    fn front(&self) -> &T {&self.data[0]}

    fn back(&self) -> &T {&self.data[self.len-1]}

    fn from(vals: &[T]) -> Self {
        let mut c = Chunk::new();
        c.data[..vals.len()].copy_from_slice(vals);
        c.len = vals.len();
        c
    }

    fn split(&mut self) -> Self {
        self.len = SPLIT_POINT;
        Chunk::from(&self.data[SPLIT_POINT..])
    }

    fn push_back(&mut self, val: T) -> Option<Self> {
        if self.len == CHUNK_SIZE {
            let mut n = self.split();
            n.push_back(val);
            Some(n)
        } else {
            self.data[self.len] = val;
            self.len += 1;
            None
        }
    }

    fn pop_back(&mut self) -> T {
        self.len -= 1;
        self.data[self.len]
    }

    fn merge_next(&mut self, other: &mut Chunk<T>) {
        let cap = cmp::min(other.len + self.len, CHUNK_SIZE);
        for i in self.len..cap { self.data[i] = other.data[i-self.len]; }
        self.len = cap;
    }

    fn push_front(&mut self, val: T) -> Option<Self> {
        if self.len == CHUNK_SIZE {
            let n = self.split();
            self.push_front(val);
            Some(n)
        } else {
            for i in (1..=self.len).rev() { self.data[i] = self.data[i-1]; }
            self.data[0] = val;
            self.len += 1;
            None
        }
    }

    fn remove(&mut self, idx: usize) -> T {
        let ret = self.data[idx];
        for i in idx..(self.len - 1) { self.data[i] = self.data[i+1] }
        self.len -= 1;
        ret
    }

    fn insert_at(&mut self, val: T, idx: usize) -> Option<Self> {
        if self.len == CHUNK_SIZE {
            let mut n = self.split();
            match idx {
                SPLIT_POINT..=CHUNK_SIZE => n.insert_at(val, idx - SPLIT_POINT),
                _ => self.insert_at(val, idx)
            };
            Some(n)
        } else {
            for i in ((idx+1)..=self.len).rev() { self.data[i] = self.data[i-1] }
            self.data[idx] = val;
            self.len += 1;
            None
        }
    }

    fn iter(&self) -> ChunkIter<'_, T> {
        ChunkIter{next: self.data.iter(), idx: 0, cap: self.len}
    }

    fn iter_mut(&mut self) -> ChunkIterMut<'_, T> {
        ChunkIterMut{next: self.data.iter_mut(), idx: 0, cap: self.len}
    }
}

struct ChunkIter<'a, T: Default + Copy> {
    next: ArrIter<'a, T>,
    idx: usize,
    cap: usize
}

impl<'a, T: Default + Copy> Iterator for ChunkIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> { 
        if self.idx < self.cap { self.idx += 1; self.next.next() } else { None }
    }
}

impl<'a, T: Default + Copy> DoubleEndedIterator for ChunkIter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.idx > 0 {self.idx -= 1; self.next.next_back() } else { None }
    }
}

struct ChunkIterMut<'a, T: Default + Copy> {
    next: ArrIterMut<'a, T>,
    idx: usize,
    cap: usize
}

impl<'a, T: Default + Copy> Iterator for ChunkIterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx <= self.cap { self.idx += 1; self.next.next() } else { None }
    }
}

impl<'a, T: Default + Copy> DoubleEndedIterator for ChunkIterMut<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.idx > 0 {self.idx -= 1; self.next.next_back() } else { None }
    }
}

struct ChunkIntoIter<T: Default + Copy> {
    chunk: Chunk<T>,
    idx: usize
}

impl<T: Default + Copy> Iterator for ChunkIntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx == self.chunk.len { None } else {
            self.idx += 1;
            Some(self.chunk.data[self.idx - 1])
        }
    }
}

impl<T: Default + Copy> IntoIterator for Chunk<T> {
    type Item = T;
    type IntoIter = ChunkIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        ChunkIntoIter{chunk: self, idx: 0}
    }
}


/// Linked list of short, fixed arrays of values.
/// Insertion via cursor is slightly more expensive than with
/// linked list but is still constant time.
#[derive(Clone)]
pub struct ChunkyList<T> 
where
    T: Copy + Default
{
    data: LinkedList<Chunk<T>>,
    len: usize
}

impl<T: Default + Copy + PartialEq> PartialEq for ChunkyList<T> {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl<T: Default + Copy + PartialEq> Eq for ChunkyList<T> {}

impl<T: Default + Copy + Ord> PartialOrd for ChunkyList<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}

impl<T: Default + Copy + Ord> Ord for ChunkyList<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<T: Default + Copy + fmt::Debug> fmt::Debug for ChunkyList<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.iter().collect::<Vec<&T>>().fmt(f)
    }
}

impl<T: Default + Copy> Default for ChunkyList<T> {
    fn default() -> Self {
        ChunkyList{data: LinkedList::new(), len: 0}
    }
}

impl<T: Default + Copy> FromIterator<T> for ChunkyList<T> {
    fn from_iter<I: IntoIterator<Item=T>>(iter: I) -> Self {
        let mut l = ChunkyList::default();
        for el in iter { l.push_back(el) }
        l
    }
}

impl<T: Default + Copy> AddAssign for ChunkyList<T> {
    fn add_assign(&mut self, mut other: Self) {
        self.append(&mut other);
    }
}

impl<T: Default + Copy> ChunkyList<T> {

    /// Returns an empty chunky list
    pub fn new() -> Self { Default::default() }

    /// Construct from an array reference
    /// 
    /// # Arguments
    /// 
    /// * `data` - array slice from which we construct out list
    /// 
    /// # Example
    /// 
    /// ```
    /// use chunky_list::ChunkyList;
    /// 
    /// let mut list = ChunkyList::from(&[1,2,3,4]);
    /// 
    /// assert_eq!(format!("{:?}", list), "[1, 2, 3, 4]")
    /// ```
    pub fn from(data: &[T]) -> Self{
        let mut l = ChunkyList::default();
        for el in data { l.push_back(*el) }
        l
    }

    /// Returns the number of items in the list
    /// 
    /// # Example
    /// 
    /// ```
    /// use chunky_list::ChunkyList;
    /// 
    /// let list = ChunkyList::from(&[1,2,3,4,5,6,7,8,9]);
    /// 
    /// assert_eq!(list.len(), 9);
    /// ```
    pub fn len(&self) -> usize { self.len }

    /// Returns true if our list is empty
    /// 
    /// # Example
    /// 
    /// ```
    /// use chunky_list::ChunkyList;
    /// 
    /// let mut list: ChunkyList<i32> = ChunkyList::new();
    /// 
    /// assert!(list.is_empty());
    /// 
    /// list.push_back(42);
    /// 
    /// assert!(!list.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Optional reference to the first element
    /// 
    /// # Example
    /// 
    /// ```
    /// use chunky_list::ChunkyList;
    /// 
    /// let mut list: ChunkyList<i32> = ChunkyList::new();
    /// 
    /// assert_eq!(list.front(), None);
    /// 
    /// list.push_back(1);
    /// list.push_back(2);
    /// list.push_back(3);
    /// 
    /// assert_eq!(list.front(), Some(&1));
    /// ```
    pub fn front(&self) -> Option<&T> {
        self.data.front().map(|c| c.front())
    }

    /// Optional reference to the last element
    /// 
    /// # Example
    /// 
    /// ```
    /// use chunky_list::ChunkyList;
    /// 
    /// let mut list: ChunkyList<i32> = ChunkyList::new();
    /// 
    /// assert_eq!(list.back(), None);
    /// 
    /// list.push_back(1);
    /// list.push_back(2);
    /// list.push_back(3);
    /// 
    /// assert_eq!(list.back(), Some(&3));
    /// ```
    pub fn back(&self) -> Option<&T> {
        self.data.back().map(|c| c.back())
    }

    /// pushes the item onto the end of the list
    /// 
    /// # Arguments
    /// 
    /// * `val` - a value to be added to the end of the list
    /// 
    /// # Example
    /// 
    /// ```
    /// use chunky_list::ChunkyList;
    /// 
    /// let mut list: ChunkyList<i32> = ChunkyList::new();
    /// 
    /// list.push_back(1);
    /// list.push_back(2);
    /// list.push_back(3);
    /// 
    /// assert_eq!(list, ChunkyList::from(&[1,2,3]));
    /// ```
    pub fn push_back(&mut self, val: T) {
        if self.len == 0 {
            self.data.push_back(Chunk::new());
        }
        if let Some(rest) = self.data.back_mut().unwrap().push_back(val) {
            self.data.push_back(rest);
        }
        self.len += 1;
    }

    /// Removes the last item from the back of the list
    /// 
    /// # Example
    /// 
    /// ```
    /// use chunky_list::ChunkyList;
    /// 
    /// let mut list = ChunkyList::from(&[1,2,3]);
    /// 
    /// assert_eq!(list.pop_back(), Some(3));
    /// assert_eq!(list.pop_back(), Some(2));
    /// assert_eq!(list.pop_back(), Some(1));
    /// 
    /// assert_eq!(list.pop_back(), None);
    /// // Will continue returning None
    /// assert_eq!(list.pop_back(), None);
    /// ```
    pub fn pop_back(&mut self) -> Option<T> {
        if self.len == 0 { return None; }
        self.len -= 1;
        let end = self.data.back_mut().unwrap();
        let ret = Some(end.pop_back());
        if end.len == 0 {
            self.data.pop_back();
        }
        ret
    }

    /// Appends second array onto the first one, emptying the second array in the process
    /// Should be faster than pushing back each element separately for larger lists
    /// 
    /// # Arguments
    /// 
    /// `other` - Other chunky list containing the same elements
    /// 
    /// # Example
    /// 
    /// ```
    /// use chunky_list::ChunkyList;
    /// 
    /// let mut s_front = ChunkyList::from(&[1,2,3,4]);
    /// let mut s_back = ChunkyList::from(&[5,6,7,8]);
    /// 
    /// let s_whole = ChunkyList::from(&[1,2,3,4,5,6,7,8]);
    /// 
    /// s_front.append(&mut s_back);
    /// 
    /// assert!(s_back.is_empty());
    /// assert_eq!(s_front, s_whole);
    /// ```
    pub fn append(&mut self, other: &mut Self) {
        if other.len() < CHUNK_SIZE / 2 {
            other.iter().for_each(|el| self.push_back(*el));
        } else {
            self.len += other.len;
            self.data.append(&mut other.data);
            other.len = 0;
        }
    }

    /// Appends second array onto the first one, without modifying the second array
    /// Should be faster than pushing back each element separately for larger lists
    /// 
    /// # Arguments
    /// 
    /// `other` - Other chunky list containing the same elements
    /// 
    /// # Example
    /// 
    /// ```
    /// use chunky_list::ChunkyList;
    /// 
    /// let mut s_front = ChunkyList::from(&[1,2,3,4]);
    /// let s_back = ChunkyList::from(&[5,6,7,8]);
    /// 
    /// let s_whole = ChunkyList::from(&[1,2,3,4,5,6,7,8]);
    /// 
    /// s_front.append_copy(&s_back);
    /// 
    /// assert!(!s_back.is_empty());
    /// assert_eq!(s_front, s_whole);
    pub fn append_copy(&mut self, other: &Self) {
        self.append(&mut other.clone());
    }

    /// Deletes all elements from the list, making it empty
    /// 
    /// # Example
    /// 
    /// ```
    /// use chunky_list::ChunkyList;
    /// 
    /// let mut s = ChunkyList::from(&[1,2,3,4]);
    /// 
    /// assert!(!s.is_empty());
    /// 
    /// s.clear();
    /// 
    /// assert!(s.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.len = 0;
        self.data.clear();
    }

    /// Returns an iterator over immutable references to list elements
    /// 
    /// # Example
    /// 
    /// ```
    /// use chunky_list::ChunkyList;
    /// 
    /// let s = ChunkyList::from(&[1,2,3]);
    /// let mut iter = s.iter();
    /// 
    /// assert_eq!(iter.next(), Some(&1));
    /// assert_eq!(iter.next(), Some(&2));
    /// assert_eq!(iter.next(), Some(&3));
    /// 
    /// // iter returns none after list
    /// assert_eq!(iter.next(), None);
    /// // continues returning none
    /// assert_eq!(iter.next(), None);
    /// ```
    pub fn iter(&self) -> Iter<T> {
        Iter::new(self.data.iter())
    }
    
    /// Returns an iterator over mutable references to list elements
    /// 
    /// # Example
    /// 
    /// ```
    /// use chunky_list::ChunkyList;
    /// 
    /// let mut s1 = ChunkyList::from(&[1,2,3]);
    /// let s2 = ChunkyList::from(&[2,4,6]);
    /// let mut iter = s1.iter_mut();
    /// iter.for_each(|el| *el *= 2);
    /// 
    /// assert_eq!(s1, s2);
    /// ```
    pub fn iter_mut(&mut self) -> IterMut<T> {
        IterMut::new(self.data.iter_mut())
    }

    /// Returns a cursor to the head of the list
    pub fn cursor_head(&self) -> Cursor<T> {
        Cursor{list_cursor: self.data.cursor_head(), idx: 0}
    }

    /// Returns a cursor to the tail of the list
    pub fn cursor_tail(&self) -> Cursor<T> {
        let tail_idx = if let Some(chunk) = self.data.back() { chunk.len-1 } else { 0 };
        Cursor{list_cursor: self.data.cursor_tail(), idx: tail_idx}
    }

    /// Returns a mutable cursor to the head of the list 
    pub fn cursor_head_mut(&mut self) -> CursorMut<T> {
        CursorMut{list_cursor: self.data.cursor_head_mut(), list_len: &mut self.len, idx: 0}
    }

    /// Returns a mutable cursor to the tail of the list
    pub fn cursor_tail_mut(&mut self) -> CursorMut<T> {
        let tail_idx = if let Some(chunk) = self.data.back() { chunk.len-1 } else { 0 };
        CursorMut{list_cursor: self.data.cursor_tail_mut(), list_len: &mut self.len, idx: tail_idx}
    }

    /// Returns the fraction of chunkspace used by data
    /// 
    /// # Example
    /// 
    /// ```
    /// use chunky_list::ChunkyList;
    /// use std::iter::FromIterator;
    /// 
    /// let s1 = ChunkyList::from_iter(0..20);
    /// assert_eq!(s1.utilization(), 0.625f32);
    pub fn utilization(&self) -> f32 {
        self.len as f32 / ((self.data.len()*CHUNK_SIZE) as f32)
    }
}



/// Immutable iterator over the chunky list
pub struct Iter<'a, T: Copy + Default> {
    list_iter: LinkedListIter<'a, Chunk<T>>,
    maybe_chunk_iter: Option<ChunkIter<'a, T>>
}

impl<'a, T: Default + Copy> Iter<'a, T> {
    fn new(list_iter: LinkedListIter<'a, Chunk<T>>) -> Self {
        let mut new_iter = Iter{list_iter, maybe_chunk_iter: None};
        new_iter.advance_chunk_iter();
        new_iter
    }

    fn advance_chunk_iter(&mut self) {
        self.maybe_chunk_iter = self.list_iter.next().map(|c| c.iter());
    }
}

impl<'a, T: Default + Copy> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        self.maybe_chunk_iter.take().and_then(|mut chunk_iter| {
            let ret = chunk_iter.next();
            match ret {
                Some(v) => {
                    self.maybe_chunk_iter = Some(chunk_iter);
                    Some(v)
                },
                None => {
                    self.advance_chunk_iter();
                    let ret = self.maybe_chunk_iter.as_mut().map(|c| c.next());
                    ret.flatten()
                }
            }
        })
    }
}

/// mutable iterator over the chunky list
pub struct IterMut<'a, T: Default + Copy> {
    list_iter: LinkedListIterMut<'a, Chunk<T>>,
    maybe_chunk_iter: Option<ChunkIterMut<'a, T>>
}

impl<'a, T: Default + Copy> IterMut<'a, T>{
    fn new(list_iter: LinkedListIterMut<'a, Chunk<T>>) -> Self {
        let mut new_iter = IterMut{list_iter, maybe_chunk_iter: None};
        new_iter.advance_chunk_iter();
        new_iter
    }
    
    fn advance_chunk_iter(&mut self) {
        self.maybe_chunk_iter = self.list_iter.next().map(|c| c.iter_mut())
    }
}

impl<'a, T: Default + Copy> Iterator for IterMut<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        self.maybe_chunk_iter.take().and_then(|mut chunk_iter| {
            let ret = chunk_iter.next();
            match ret {
                Some(v) => {
                    self.maybe_chunk_iter = Some(chunk_iter);
                    Some(v)
                },
                None => {
                    self.advance_chunk_iter();
                    let ret = self.maybe_chunk_iter.as_mut().map(|c| c.next());
                    ret.flatten()
                }
            }
        })
    }
}


// Iterator that consumes
pub struct IntoIter<T: Default + Copy> {
    list_iter: LinkedListIntoIter<Chunk<T>>,
    maybe_chunk_iter: Option<ChunkIntoIter<T>>
}

impl<T: Default + Copy> IntoIter<T>{
    fn new(list_iter: LinkedListIntoIter<Chunk<T>>) -> Self {
        let mut new_iter = IntoIter{list_iter, maybe_chunk_iter: None};
        new_iter.advance_chunk_iter();
        new_iter
    }

    fn advance_chunk_iter(&mut self) {
        self.maybe_chunk_iter = self.list_iter.next().map(|c| c.into_iter());
    }
}

impl<T: Default + Copy> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.maybe_chunk_iter.take().and_then(|mut chunk_iter| {
            let ret = chunk_iter.next();
            match ret {
                Some(v) => {
                    self.maybe_chunk_iter = Some(chunk_iter);
                    Some(v)
                }
                None => {
                    self.advance_chunk_iter();
                    let ret = self.maybe_chunk_iter.as_mut().map(|c| c.next());
                    ret.flatten()
                }
            }
        })
    }
}

impl<T: Default + Copy> IntoIterator for ChunkyList<T> {
    type Item = T;
    type IntoIter = IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter::new(self.data.into_iter())
    }
}

/// Cursor that seeks back and forth aacross the datastructure
/// Cursor also wraps off the end, with a None element sitting after end and before begin
pub struct Cursor<'a, T> 
where T: Default + Copy
{
    list_cursor: LinkedListCursor<'a, Chunk<T>>,
    idx: usize,
}

impl<'a, T: Default + Copy> Cursor<'a, T> {

    /// Moves the cursor to the next spot in the list
    /// 
    /// # Example
    /// 
    /// ```
    /// use chunky_list::ChunkyList;
    /// use std::iter::FromIterator;
    /// 
    /// let list = ChunkyList::from_iter(1..20);
    /// 
    /// let mut cursor = list.cursor_head();
    /// cursor.seek_next();
    /// cursor.seek_next();
    /// cursor.seek_next();
    /// cursor.seek_next();
    /// cursor.seek_next();
    /// cursor.seek_next();
    /// 
    /// assert_eq!(cursor.as_ref(), Some(&7));
    pub fn seek_next(&mut self) {
        match self.list_cursor.as_ref() {
            None => self.list_cursor.seek_next(),
            Some(chunk) => {
                self.idx += 1;
                if self.idx == chunk.len {
                    self.idx = 0;
                    self.list_cursor.seek_next();
                }
            }
        }
    }

    /// Moves the cursor to the previous spot in the list
    /// 
    /// # Example
    /// 
    /// ```
    /// use chunky_list::ChunkyList;
    /// use std::iter::FromIterator;
    /// 
    /// let list = ChunkyList::from_iter(1..20);
    /// 
    /// let mut cursor = list.cursor_head(); // 1
    /// assert_eq!(cursor.as_ref(), Some(&1));
    /// 
    /// cursor.seek_prev(); // None
    /// assert_eq!(cursor.as_ref(), None);
    /// 
    /// cursor.seek_prev(); // 19
    /// assert_eq!(cursor.as_ref(), Some(&19));
    /// 
    /// cursor.seek_prev(); // 18
    /// assert_eq!(cursor.as_ref(), Some(&18));
    /// 
    /// cursor.seek_prev(); // 17
    /// assert_eq!(cursor.as_ref(), Some(&17));
    /// 
    /// cursor.seek_prev(); // 16
    /// assert_eq!(cursor.as_ref(), Some(&16));
    /// 
    /// cursor.seek_prev(); // 15
    /// assert_eq!(cursor.as_ref(), Some(&15));
    pub fn seek_prev(&mut self) {
        match self.list_cursor.as_ref() {
            None => { 
                self.list_cursor.seek_prev();
                self.idx = self.list_cursor.as_ref().map_or(0, |chunk| chunk.len-1);
            }, 
            Some(_) => {
                if self.idx == 0 {
                    self.list_cursor.seek_prev();
                    self.idx = self.list_cursor.as_ref().map_or(0, |chunk| chunk.len);
                } else { self.idx -= 1; }
            }
        }
    }

    /// Returns a reference to the value pointed to by the cursor
    pub fn as_ref(&self) -> Option<&T> {
        self.list_cursor.as_ref().map(|chunk| &chunk.data[self.idx])
    }
}

pub struct CursorMut<'a, T>
where T: Default + Copy
{
    list_cursor: LinkedListCursorMut<'a, Chunk<T>>,
    list_len: &'a mut usize,
    idx: usize
}

impl<'a, T: Default + Copy> CursorMut<'a, T> {

    /// Moves the cursor to the next spot in the list
    /// 
    /// # Example
    /// 
    /// ```
    /// use chunky_list::ChunkyList;
    /// use std::iter::FromIterator;
    /// 
    /// let mut list = ChunkyList::from_iter(1..20);
    /// 
    /// let mut cursor = list.cursor_head_mut();
    /// cursor.seek_next();
    /// cursor.seek_next();
    /// cursor.seek_next();
    /// cursor.seek_next();
    /// cursor.seek_next();
    /// cursor.seek_next();
    /// assert_eq!(cursor.as_ref(), Some(&7));
    /// cursor.as_mut().map(|num| *num -= 7 );
    /// assert_eq!(cursor.as_ref(), Some(&0));
    pub fn seek_next(&mut self) {
        match self.list_cursor.as_ref() {
            None => self.list_cursor.seek_next(),
            Some(chunk) => {
                self.idx += 1;
                if self.idx == chunk.len {
                    self.idx = 0;
                    self.list_cursor.seek_next();
                }
            }
        }
    }

    /// Moves the cursor to the previous spot in the list
    /// 
    /// # Example
    /// 
    /// ```
    /// use chunky_list::ChunkyList;
    /// use std::iter::FromIterator;
    /// 
    /// let mut list = ChunkyList::from_iter(1..20);
    /// 
    /// let mut cursor = list.cursor_head_mut(); // 1
    /// assert_eq!(cursor.as_ref(), Some(&1));
    /// 
    /// cursor.seek_prev(); // None
    /// assert_eq!(cursor.as_ref(), None);
    /// 
    /// cursor.seek_prev(); // 19
    /// assert_eq!(cursor.as_ref(), Some(&19));
    /// 
    /// cursor.seek_prev(); // 18
    /// assert_eq!(cursor.as_ref(), Some(&18));
    /// 
    /// cursor.seek_prev(); // 17
    /// assert_eq!(cursor.as_ref(), Some(&17));
    /// 
    /// cursor.seek_prev(); // 16
    /// assert_eq!(cursor.as_ref(), Some(&16));
    /// 
    /// cursor.seek_prev(); // 15
    /// assert_eq!(cursor.as_ref(), Some(&15));
    /// 
    /// cursor.as_mut().map(|num| *num -= 10);
    /// assert_eq!(cursor.as_ref(), Some(&5));
    /// ```
    pub fn seek_prev(&mut self) {
        match self.list_cursor.as_ref() {
            None => { 
                self.list_cursor.seek_prev();
                self.idx = self.list_cursor.as_ref().map_or(0, |chunk| chunk.len-1);
            }, 
            Some(_) => {
                if self.idx == 0 {
                    self.list_cursor.seek_prev();
                    self.idx = self.list_cursor.as_ref().map_or(0, |chunk| chunk.len);
                } else { self.idx -= 1; }
            }
        }
    }

    /// Returns a reference to the value pointed to by the cursor
    pub fn as_ref(&self) -> Option<&T> {
        self.list_cursor.as_ref().map(|chunk| &chunk.data[self.idx])
    }

    /// Returns a reference to the value pointed to by the cursor
    pub fn as_mut(&self) -> Option<&mut T> {
        self.list_cursor.as_mut().map(|chunk| &mut chunk.data[self.idx])
    }


    /// Inserts an element after the cursor in the chunky list
    /// 
    /// # Arguments
    /// 
    /// `val` - The value to insert into the array
    /// 
    /// # Example
    /// 
    /// ```
    /// use chunky_list::ChunkyList;
    /// use std::iter::FromIterator;
    /// 
    /// let mut list = ChunkyList::new();
    /// let mut final_list = ChunkyList::from_iter(0..100);
    /// 
    /// for i in 0..100 { if i%3 != 0 { list.push_back(i) } }
    /// 
    /// println!("{:?}", list);
    /// 
    /// let mut cursor = list.cursor_head_mut();
    /// cursor.seek_prev();
    /// 
    /// for i in (0..100).step_by(3) {
    ///     println!("{} {:?}", i, cursor.as_ref());
    ///     cursor.insert_after(i);
    ///     cursor.seek_next();
    ///     cursor.seek_next();
    ///     cursor.seek_next();
    /// }
    /// 
    /// assert_eq!(list, final_list);
    /// ```
    pub fn insert_after(&mut self, val: T) {
        match self.list_cursor.as_mut() {
            None => {
                self.seek_next();
                match self.list_cursor.as_mut() {
                    None => { // Totally Empty List
                        let mut chunk = Chunk::new();
                        chunk.push_back(val);
                        self.list_cursor.insert_before(chunk);
                    },
                    Some(chunk) => { // Was on None Element
                        if let Some(new_chunk) = chunk.push_front(val) {
                            self.list_cursor.insert_after(new_chunk);
                        }
                        self.list_cursor.seek_prev();
                    }
                }
            },
            Some(chunk) => {

                if let Some(new_chunk) = chunk.insert_at(val, self.idx+1) {
                    self.list_cursor.insert_after(new_chunk);
                    if self.idx >= SPLIT_POINT { self.list_cursor.seek_next(); self.idx -= SPLIT_POINT; }
                }
            }
        }
        *self.list_len += 1;
    }

    /// Inserts an element before the cursor in the chunky list
    /// 
    /// # Arguments
    /// 
    /// `val` - The value to insert into the array
    /// 
    /// # Examples
    /// 
    /// ```
    /// 
    /// ```
    /// use chunky_list::ChunkyList;
    /// use std::iter::FromIterator;
    /// 
    /// let mut list = ChunkyList::new();
    /// let mut final_list = ChunkyList::from_iter(0..100);
    /// 
    /// for i in 0..100 { if i%3 != 0 { list.push_back(i) } }
    /// 
    /// println!("{:?}", list);
    /// 
    /// let mut cursor = list.cursor_head_mut();
    /// 
    /// for i in (0..100).step_by(3) {
    ///     cursor.insert_before(i);
    ///     cursor.seek_next();
    ///     cursor.seek_next();
    /// }
    /// 
    /// assert_eq!(list, final_list);
    /// ```
    pub fn insert_before(&mut self, val: T) {
        match self.list_cursor.as_mut() {
            None => {
                self.seek_prev();
                match self.list_cursor.as_mut() {
                    None => { // Totally Empty List
                        let mut chunk = Chunk::new();
                        chunk.push_back(val);
                        self.list_cursor.insert_after(chunk);
                    },
                    Some(chunk) => { // Was on None element
                        if let Some(new_chunk) = chunk.push_back(val) {
                            self.list_cursor.insert_after(new_chunk);
                            self.list_cursor.seek_next()
                        }
                        self.list_cursor.seek_next();
                        self.idx = 0;
                    }
                }
            },
            Some(chunk) => {
                self.idx += 1;
                if let Some(new_chunk) = chunk.insert_at(val, self.idx-1) {
                    self.list_cursor.insert_after(new_chunk);
                    if self.idx >= SPLIT_POINT {self.list_cursor.seek_next(); self.idx -= SPLIT_POINT; }
                }
            }
        }
        *self.list_len += 1;
    }

    fn merge_next(&mut self) {
        if self.list_cursor.as_ref().is_none() { self.list_cursor.seek_next() } else {
            let mut my_chunk = self.list_cursor.remove().unwrap();
            if let Some(mut next_chunk) = self.list_cursor.remove() {
                my_chunk.merge_next(&mut next_chunk);
            }
            self.list_cursor.insert_before(my_chunk);
            self.list_cursor.seek_prev();
        }
    }

    fn get_prev_len(&mut self) -> Option<usize> {
        self.list_cursor.seek_prev();
        let ret = self.list_cursor.as_ref().map(|c| c.len);
        self.list_cursor.seek_next();
        ret
    }

    fn get_next_len(&mut self) -> Option<usize> {
        self.list_cursor.seek_next();
        let ret = self.list_cursor.as_ref().map(|c| c.len);
        self.list_cursor.seek_prev();
        ret
    }

    /// Removes the element currently pointed to by the cursor if it exists.
    /// Returns None if pointing off end of list
    /// 
    /// # Example
    /// 
    /// ```
    /// use chunky_list::ChunkyList;
    /// use std::iter::FromIterator;
    /// 
    /// let mut list = ChunkyList::from_iter(1..20);
    /// let mut cursor = list.cursor_head_mut();
    /// for _ in 0..10 {
    ///     cursor.seek_next();
    /// }
    /// for i in 11..20 {
    ///     let temp = cursor.remove();
    ///     println!("{:?}", temp);
    ///     assert_eq!(temp, Some(i));
    /// }
    /// //assert_eq!(cursor.remove(), None);
    /// cursor.seek_next();
    /// 
    /// for i in 1..11 {
    ///     let temp = cursor.remove();
    ///     println!("{:?}", temp);
    ///     assert_eq!(temp, Some(i));
    /// }
    /// assert_eq!(cursor.remove(), None);
    /// assert!(list.is_empty())
    /// ```
    pub fn remove(&mut self) -> Option<T> {
        self.list_cursor.as_ref()?;
        
        let ret = self.list_cursor.as_mut().unwrap().remove(self.idx);
        *self.list_len -= 1;

        let prev_len = self.get_prev_len().unwrap_or(CHUNK_SIZE);
        let next_len = self.get_next_len().unwrap_or(CHUNK_SIZE);
        let curr_len = self.list_cursor.as_ref().unwrap().len;

        if curr_len == 0 {
            self.list_cursor.remove();
            self.idx = 0;
        } else if curr_len + next_len <= CHUNK_SIZE {
            self.merge_next();
        } else if curr_len + prev_len <= CHUNK_SIZE {
            self.list_cursor.seek_prev();
            self.idx += prev_len;
            self.merge_next();
        } else if self.idx >= curr_len {
            self.idx -= curr_len;
            self.list_cursor.seek_next();
        }

        Some(ret)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_init() {
        let mut s = ChunkyList::default();
        s.push_back(1);
        s.push_back(2);
        s.push_back(3);
        let mut i = s.iter();
        assert_eq!(i.next(), Some(&1));
        assert_eq!(i.next(), Some(&2));
        assert_eq!(i.next(), Some(&3));

        assert_eq!(i.next(), None);
    }

    #[test]
    fn from_vec_init() {
        let s = ChunkyList::from(&[1,2,3]);
        let mut i = s.iter();
        assert_eq!(i.next(), Some(&1));
        assert_eq!(i.next(), Some(&2));
        assert_eq!(i.next(), Some(&3));

        assert_eq!(i.next(), None);
    }

    #[test]
    fn long_iter() {
        let s = ChunkyList::from_iter(1..20);
        let mut iter = s.iter();
        for i in 1..20 {
            assert_eq!(iter.next(), Some(&i));
        }
    }

    #[test]
    fn front_back() {
        let s = ChunkyList::from(&[1,2,3]);
        assert_eq!(s.front(), Some(&1));
        assert_eq!(s.back(), Some(&3))
    }

    #[test]
    fn front_back_empty() {
        let s: ChunkyList<i32> = ChunkyList::from(&[]);
        assert_eq!(s.front(), None);
        assert_eq!(s.back(), None);
    }

    #[test]
    fn front_back_one() {
        let s: ChunkyList<i32> = ChunkyList::from(&[1]);
        assert_eq!(s.front(), Some(&1));
        assert_eq!(s.back(), Some(&1));
    }

    #[test]
    fn eq_test() {
        let s1 = ChunkyList::from(&[1,2,3]);
        let s2 = ChunkyList::from(&[1,2,3]);
        let s3 = ChunkyList::from(&[1,2,4,3]);
        let s4 = ChunkyList::from(&[1,2,3,4]);

        assert_eq!(s1, s2);

        assert_ne!(s1, s3);
        assert_ne!(s3, s2);

        assert_ne!(s1, s4);
        assert_ne!(s4, s2);
    }

    #[test]
    fn it_mut_test() {
        let mut s1 = ChunkyList::from_iter(0..20);
        let s2 = ChunkyList::from_iter(10..30);

        assert_ne!(s1, s2);

        for el in s1.iter_mut() {
            *el += 10;
        }

        assert_eq!(s1, s2);
    }

    #[test]
    fn into_it_test() {
        let s = ChunkyList::from_iter(0..20);
        let total: i32 = s.into_iter().sum();
        assert_eq!(total, 190);
    }

    #[test]
    fn cmp_test() {
        let s1 = ChunkyList::from(b"hello world");
        let s2 = ChunkyList::from(b"Hello world");
        let s3 = ChunkyList::from(b"hello world!");
        let s4 = ChunkyList::from(b"Hello world!");

        assert!(s1 > s2);
        assert!(s2 < s1);
        
        assert!(s1 < s3);
        assert!(s3 > s1);

        assert!(s1 > s4);
        assert!(s4 < s1);

        assert!(s2 < s3);
        assert!(s3 > s2);

        assert!(s2 < s4);
        assert!(s4 > s2);
        
        assert!(s3 > s4);
        assert!(s4 < s3);
    }

    #[test]
    fn plus_eq() {
        let mut s_front = ChunkyList::from_iter(1i32..10);
        let s_back = ChunkyList::from_iter(10i32..20);
        let s_total: Vec<i32> = (1..20).collect();

        s_front += s_back;

        println!("{:?}",s_front);

        let s_front: Vec<i32> = s_front.into_iter().collect();

        println!("{:?}",s_front);

        assert_eq!(s_front, s_total);
    }
}