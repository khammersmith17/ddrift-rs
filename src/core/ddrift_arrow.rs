use crate::core::bin_edges::{CategoricalBinEdges, ContinuousBinEdges};
use crate::ddrift_arrow::slice_impl::SliceImpl;
use arrow::buffer::NullBuffer;
use num_traits::float::Float;
use std::hash::Hash;

pub(crate) fn compute_bins_arrow_string_slice<
    'slice,
    'a,
    S: SliceImpl<&'slice str> + Send + Sync,
>(
    slice: &S,
    bin_edges: &'a CategoricalBinEdges<String>,
) -> (Vec<f64>, f64) {
    let thread_count = crate::constants::get_thread_count(slice.len());
    if thread_count > 1 {
        crate::core::opt::arrow_opt::parallel_approx_string_slice(slice, bin_edges, thread_count)
    } else {
        single_thread::string_slice_seq(slice, bin_edges)
    }
}

pub(crate) fn compute_bins_arrow_bool_slice<'a, S: SliceImpl<bool> + Send + Sync>(
    slice: &S,
    bin_edges: &CategoricalBinEdges<bool>,
) -> (Vec<f64>, f64) {
    let thread_count = crate::constants::get_thread_count(slice.len());
    if thread_count > 1 {
        crate::core::opt::arrow_opt::parallel_approx_boolean_slice(slice, bin_edges, thread_count)
    } else {
        single_thread::bool_slice_seq(slice, bin_edges)
    }
}

pub(crate) fn compute_bins_continuous<T: Float + Send + Sync>(
    dataset: &[T],
    bins: &ContinuousBinEdges<T>,
    null_buffer: Option<&NullBuffer>,
) -> (Vec<f64>, f64) {
    let thread_count = crate::constants::get_thread_count(dataset.len());
    if thread_count > 1 {
        crate::core::opt::arrow_opt::parallel_approx_arrow_cont(
            dataset,
            bins,
            null_buffer,
            thread_count,
        )
    } else {
        single_thread::cont_seq(dataset, bins, null_buffer)
    }
}

pub(crate) fn compute_bins_categorical<T: Hash + Ord + Clone + Send + Sync>(
    dataset: &[T],
    bins: &CategoricalBinEdges<T>,
    null_buffer: Option<&NullBuffer>,
) -> (Vec<f64>, f64) {
    let thread_count = crate::constants::get_thread_count(dataset.len());
    if thread_count > 1 {
        crate::core::opt::arrow_opt::parallel_approx_arrow_cat(
            dataset,
            bins,
            null_buffer,
            thread_count,
        )
    } else {
        single_thread::cat_seq(dataset, bins, null_buffer)
    }
}

/// Entry points into the single threaded computation path. Used when the dataset is small enough
/// that threading overhead is not justified (<10k items).
mod single_thread {
    use super::Hash;
    pub(super) fn cont_seq<T: super::Float>(
        dataset: &[T],
        bins: &super::ContinuousBinEdges<T>,
        null_buffer: Option<&super::NullBuffer>,
    ) -> (Vec<f64>, f64) {
        if let Some(null_buff) = null_buffer {
            cont_seq_null(dataset, bins, null_buff)
        } else {
            (
                crate::core::compute_dataset_from_bins_continuous_seq(dataset, bins),
                0_f64,
            )
        }
    }

    fn cont_seq_null<T: super::Float>(
        dataset: &[T],
        bins: &super::ContinuousBinEdges<T>,
        null_buffer: &super::NullBuffer,
    ) -> (Vec<f64>, f64) {
        let n_bins = bins.n_bins();
        let mut hist = vec![0_f64; n_bins];
        let mut null_count = 0_f64;

        dataset.iter().enumerate().for_each(|(i, ex)| {
            if null_buffer.is_valid(i) {
                hist[bins.resolve_bin(*ex)] += 1_f64;
            } else {
                null_count += 1_f64;
            }
        });
        (hist, null_count)
    }

    pub(super) fn cat_seq<T: Hash + Ord + Clone>(
        dataset: &[T],
        bins: &super::CategoricalBinEdges<T>,
        null_buffer: Option<&super::NullBuffer>,
    ) -> (Vec<f64>, f64) {
        if let Some(null_buff) = null_buffer {
            cat_seq_null(dataset, bins, null_buff)
        } else {
            (
                crate::core::compute_dataset_from_bins_categorical(dataset, bins),
                0_f64,
            )
        }
    }

    fn cat_seq_null<T: Hash + Ord + Clone>(
        dataset: &[T],
        bins: &super::CategoricalBinEdges<T>,
        null_buffer: &super::NullBuffer,
    ) -> (Vec<f64>, f64) {
        let n_bins = bins.n_bins();
        let mut hist = vec![0_f64; n_bins];
        let mut null_count = 0_f64;

        dataset.iter().enumerate().for_each(|(i, ex)| {
            if null_buffer.is_valid(i) {
                hist[bins.resolve_bin(ex)] += 1_f64;
            } else {
                null_count += 1_f64;
            }
        });
        (hist, null_count)
    }

    pub(super) fn string_slice_seq<'slice, 'a, S: super::SliceImpl<&'slice str>>(
        slice: &S,
        bin_edges: &'a super::CategoricalBinEdges<String>,
    ) -> (Vec<f64>, f64) {
        let mut hist = vec![0_f64; bin_edges.n_bins()];
        let mut null_count = 0_f64;
        for idx in slice.index_range() {
            if let Some(ex) = slice.get(idx) {
                hist[bin_edges.resolve_bin(ex)] += 1_f64;
            } else {
                null_count += 1_f64;
            }
        }
        (hist, null_count)
    }

    pub(super) fn bool_slice_seq<'a, S: super::SliceImpl<bool>>(
        slice: &S,
        bin_edges: &'a super::CategoricalBinEdges<bool>,
    ) -> (Vec<f64>, f64) {
        let mut hist = vec![0_f64; bin_edges.n_bins()];
        let mut null_count = 0_f64;
        for idx in slice.index_range() {
            if let Some(ex) = slice.get(idx) {
                hist[bin_edges.resolve_bin(&ex)] += 1_f64;
            } else {
                null_count += 1_f64;
            }
        }
        (hist, null_count)
    }
}
