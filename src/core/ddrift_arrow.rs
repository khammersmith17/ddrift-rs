use crate::core::bin_edges::{CategoricalBinEdges, ContinuousBinEdges};
use arrow::buffer::NullBuffer;
use num_traits::float::Float;
use std::hash::Hash;

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
}
