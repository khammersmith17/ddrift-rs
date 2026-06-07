use arrow::array::Array;
use std::ops::Range;
use std::vec::IntoIter;

/// This trait allows for a slice like representation of arrow like String storage representations,
/// where all strings are stored in a contiguous u8 buffer and offsets of the strings are stored
/// along side. This allows for distributed computation by abstracting out the logic that provides
/// a typed refernece to the chunk within the larger slab.
pub trait SliceImpl<T> {
    fn get(&self, i: usize) -> Option<T>;
    fn len(&self) -> usize;
    fn chunk_indexes(&self, mut num_chunks: usize) -> IntoIter<Range<usize>> {
        num_chunks = num_chunks.min(self.len());
        let mut range_chunks: Vec<Range<usize>> = Vec::with_capacity(num_chunks);
        let mut start = 0_usize;
        let len = self.len();
        let step = (len as f64 / num_chunks as f64).ceil() as usize;
        for _ in 0..num_chunks {
            if start > len {
                break;
            }
            range_chunks.push(start..(start + step).min(len));
            start += step;
        }
        range_chunks.into_iter()
    }
    fn index_range(&self) -> Range<usize> {
        0..self.len()
    }
    fn is_empty(&self) -> bool {
        self.len() == 0_usize
    }
}

/// Type to implement an alternative string view that allows for parallel chunking. Abstracts the
/// underlying representation of something like `[arrow::array::StringArray]` into a slice like
/// view, allowing for
pub struct StringSlice32<'a> {
    buffer: &'a [u8],
    offsets: &'a [i32],
    null_buffer: Option<&'a arrow::buffer::NullBuffer>,
}

impl<'a> StringSlice32<'a> {
    pub(crate) fn from_array(array: &'a arrow::array::StringArray) -> StringSlice32<'a> {
        let buffer = array.values().as_slice();
        let offsets = array.value_offsets();
        let null_buffer = array.nulls();
        StringSlice32 {
            buffer,
            offsets,
            null_buffer,
        }
    }

    fn offset_len(&self) -> usize {
        self.offsets.len()
    }

    fn is_null(&self, i: usize) -> bool {
        if let Some(buffer) = self.null_buffer {
            buffer.is_null(i)
        } else {
            false
        }
    }
}

impl<'a> SliceImpl<&'a str> for StringSlice32<'a> {
    fn len(&self) -> usize {
        self.offset_len() - 1_usize
    }

    fn get(&self, i: usize) -> Option<&'a str> {
        let offset_len = self.offset_len();
        if i + 1 >= offset_len || self.is_null(i) {
            return None;
        }

        let start = self.offsets[i] as usize;
        let end = self.offsets[i + 1] as usize;

        // SAFETY: Safety comes from upstream arrow implementation.
        // Only constructed from safe utf8 representation.
        return Some(unsafe { std::str::from_utf8_unchecked(&self.buffer[start..end]) });
    }
}

pub struct BooleanSlice<'a> {
    buffer: &'a arrow::buffer::BooleanBuffer,
}

impl<'a> BooleanSlice<'a> {
    pub(crate) fn from_array(array: &'a arrow::array::BooleanArray) -> BooleanSlice<'a> {
        BooleanSlice {
            buffer: array.values(),
        }
    }
}

impl<'a> SliceImpl<bool> for BooleanSlice<'a> {
    fn len(&self) -> usize {
        self.buffer.len()
    }

    fn get(&self, i: usize) -> Option<bool> {
        if i >= self.buffer.len() {
            return None;
        }
        // SAFETY: bounds checked above, buffer.value handles bit offset internally.
        Some(self.buffer.value(i))
    }
}

pub struct StringSlice64<'a> {
    buffer: &'a [u8],
    offsets: &'a [i64],
    null_buffer: Option<&'a arrow::buffer::NullBuffer>,
}

impl<'a> StringSlice64<'a> {
    pub(crate) fn from_array(array: &'a arrow::array::LargeStringArray) -> StringSlice64<'a> {
        let buffer = array.values().as_slice();
        let offsets = array.value_offsets();
        let null_buffer = array.nulls();
        StringSlice64 {
            buffer,
            offsets,
            null_buffer,
        }
    }

    fn is_null(&self, i: usize) -> bool {
        if let Some(buffer) = self.null_buffer {
            buffer.is_null(i)
        } else {
            false
        }
    }

    fn offset_len(&self) -> usize {
        self.offsets.len()
    }
}

impl<'a> SliceImpl<&'a str> for StringSlice64<'a> {
    fn len(&self) -> usize {
        self.offset_len() - 1
    }
    fn get(&self, i: usize) -> Option<&'a str> {
        let offset_len = self.offset_len();
        if i + 1 >= offset_len || self.is_null(i) {
            return None;
        }

        let start = self.offsets[i] as usize;
        let end = self.offsets[i + 1] as usize;

        // SAFETY: Safety comes from upstream arrow implementation.
        // Only constructed from safe utf8 representation.
        return Some(unsafe { std::str::from_utf8_unchecked(&self.buffer[start..end]) });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BooleanArray, LargeStringArray, StringArray};

    #[test]
    fn string32_get_returns_correct_values() {
        let array = StringArray::from(vec!["foo", "bar", "baz"]);
        let slice = StringSlice32::from_array(&array);
        assert_eq!(slice.get(0), Some("foo"));
        assert_eq!(slice.get(1), Some("bar"));
        assert_eq!(slice.get(2), Some("baz"));
    }

    #[test]
    fn string32_get_out_of_bounds_returns_none() {
        let array = StringArray::from(vec!["foo", "bar"]);
        let slice = StringSlice32::from_array(&array);
        assert_eq!(slice.get(2), None);
        assert_eq!(slice.get(100), None);
    }

    #[test]
    fn string32_len_matches_array() {
        let array = StringArray::from(vec!["a", "bb", "ccc"]);
        let slice = StringSlice32::from_array(&array);
        assert_eq!(slice.len(), 3);
    }

    #[test]
    fn string32_empty_string_elements() {
        let array = StringArray::from(vec!["", "x", ""]);
        let slice = StringSlice32::from_array(&array);
        assert_eq!(slice.get(0), Some(""));
        assert_eq!(slice.get(1), Some("x"));
        assert_eq!(slice.get(2), Some(""));
    }

    #[test]
    fn string32_chunks_cover_all_indexes() {
        let array = StringArray::from(vec!["a", "b", "c", "d", "e", "f", "g"]);
        let slice = StringSlice32::from_array(&array);
        let chunks: Vec<Range<usize>> = slice.chunk_indexes(3).collect();
        let mut covered: Vec<usize> = chunks.iter().flat_map(|r| r.clone()).collect();
        covered.sort();
        assert_eq!(covered, vec![0, 1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn string32_chunks_are_non_overlapping() {
        let array = StringArray::from(vec!["a", "b", "c", "d", "e"]);
        let slice = StringSlice32::from_array(&array);
        let chunks: Vec<Range<usize>> = slice.chunk_indexes(3).collect();
        for i in 0..chunks.len() {
            for j in (i + 1)..chunks.len() {
                let overlap = chunks[i].clone().any(|x| chunks[j].contains(&x));
                assert!(!overlap, "chunks {i} and {j} overlap");
            }
        }
    }

    #[test]
    fn string32_more_chunks_than_elements_returns_at_most_len_chunks() {
        let array = StringArray::from(vec!["a", "b"]);
        let slice = StringSlice32::from_array(&array);
        let chunks: Vec<Range<usize>> = slice.chunk_indexes(10).collect();
        assert!(chunks.len() <= 2);
        assert!(chunks.iter().all(|r| !r.is_empty()));
    }

    #[test]
    fn string64_get_returns_correct_values() {
        let array = LargeStringArray::from(vec!["hello", "world"]);
        let slice = StringSlice64::from_array(&array);
        assert_eq!(slice.get(0), Some("hello"));
        assert_eq!(slice.get(1), Some("world"));
    }

    #[test]
    fn string64_get_out_of_bounds_returns_none() {
        let array = LargeStringArray::from(vec!["only"]);
        let slice = StringSlice64::from_array(&array);
        assert_eq!(slice.get(1), None);
    }

    #[test]
    fn string64_len_matches_array() {
        let array = LargeStringArray::from(vec!["x", "y", "z", "w"]);
        let slice = StringSlice64::from_array(&array);
        assert_eq!(slice.len(), 4);
    }

    #[test]
    fn string64_chunks_cover_all_indexes() {
        let array = LargeStringArray::from(vec!["a", "b", "c", "d", "e"]);
        let slice = StringSlice64::from_array(&array);
        let chunks: Vec<Range<usize>> = slice.chunk_indexes(2).collect();
        let mut covered: Vec<usize> = chunks.iter().flat_map(|r| r.clone()).collect();
        covered.sort();
        assert_eq!(covered, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn boolean_get_returns_correct_values() {
        let array = BooleanArray::from(vec![true, false, true, true, false]);
        let slice = BooleanSlice::from_array(&array);
        assert_eq!(slice.get(0), Some(true));
        assert_eq!(slice.get(1), Some(false));
        assert_eq!(slice.get(2), Some(true));
        assert_eq!(slice.get(3), Some(true));
        assert_eq!(slice.get(4), Some(false));
    }

    #[test]
    fn boolean_get_out_of_bounds_returns_none() {
        let array = BooleanArray::from(vec![true, false]);
        let slice = BooleanSlice::from_array(&array);
        assert_eq!(slice.get(2), None);
        assert_eq!(slice.get(100), None);
    }

    #[test]
    fn boolean_len_matches_array() {
        let array = BooleanArray::from(vec![true, false, true]);
        let slice = BooleanSlice::from_array(&array);
        assert_eq!(slice.len(), 3);
    }

    #[test]
    fn boolean_chunks_cover_all_indexes() {
        let array = BooleanArray::from(vec![true, false, true, false, true, false]);
        let slice = BooleanSlice::from_array(&array);
        let chunks: Vec<Range<usize>> = slice.chunk_indexes(3).collect();
        let mut covered: Vec<usize> = chunks.iter().flat_map(|r| r.clone()).collect();
        covered.sort();
        assert_eq!(covered, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn boolean_chunks_values_accessible_via_get() {
        let values = vec![true, false, false, true, true, false, true, false];
        let array = BooleanArray::from(values.clone());
        let slice = BooleanSlice::from_array(&array);
        for chunk in slice.chunk_indexes(3) {
            for i in chunk {
                assert_eq!(slice.get(i), Some(values[i]));
            }
        }
    }
}
