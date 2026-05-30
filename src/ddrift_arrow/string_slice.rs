use std::ops::Range;
use std::vec::IntoIter;

/// Type to implement an alternative string view that allows for parallel chunking.
pub(crate) struct StringSlice32<'a> {
    buffer: &'a [u8],
    offsets: &'a [i32],
}

impl<'a> StringSlice32<'a> {
    pub(crate) fn from_array(array: &'a arrow::array::StringArray) -> StringSlice32<'a> {
        let buffer = array.values().as_slice();
        let offsets = array.value_offsets();
        StringSlice32 { buffer, offsets }
    }

    fn offset_len(&self) -> usize {
        self.offsets.len()
    }

    fn len(&self) -> usize {
        self.offset_len() - 1_usize
    }

    pub(crate) fn get(&self, i: usize) -> Option<&str> {
        let offset_len = self.offset_len();
        if i + 1 >= offset_len {
            return None;
        }

        let start = self.offsets[i] as usize;
        let end = self.offsets[i + 1] as usize;

        // SAFETY: Safety comes from upstream arrow implementation.
        // Only constructed from safe utf8 representation.
        return Some(unsafe { std::str::from_utf8_unchecked(&self.buffer[start..end]) });
    }

    pub(crate) fn chunk_indexes(&self, num_chunks: usize) -> IntoIter<Range<usize>> {
        let mut range_chunks: Vec<Range<usize>> = Vec::with_capacity(num_chunks);
        let mut start = 0_usize;
        let len = self.len();
        let step = ((len + 1) as f64 / num_chunks as f64).ceil() as usize;
        for _ in 0..num_chunks {
            range_chunks.push(start..(start + step).min(len));
            start += step;
        }
        range_chunks.into_iter()
    }
}

pub(crate) struct StringSlice64<'a> {
    buffer: &'a [u8],
    offsets: &'a [i64],
}

impl<'a> StringSlice64<'a> {
    pub(crate) fn from_array(array: &'a arrow::array::LargeStringArray) -> StringSlice64<'a> {
        let buffer = array.values().as_slice();
        let offsets = array.value_offsets();
        StringSlice64 { buffer, offsets }
    }

    fn offset_len(&self) -> usize {
        self.offsets.len()
    }

    pub(crate) fn get(&self, i: usize) -> Option<&str> {
        let offset_len = self.offset_len();
        if i + 1 >= offset_len {
            return None;
        }

        let start = self.offsets[i] as usize;
        let end = self.offsets[i + 1] as usize;

        // SAFETY: Safety comes from upstream arrow implementation.
        // Only constructed from safe utf8 representation.
        return Some(unsafe { std::str::from_utf8_unchecked(&self.buffer[start..end]) });
    }
}
