use super::{CategoricalBinEdges, ContinuousBinEdges};
use crate::constants::get_thread_count;

/*
* Methods to approximate the distribution histogram across threads.
* Derive an approximate number of threads.
* Compute thread local distribution.
* Reduce local element wise count into global bins by folding the local bins and null_count when
* relevant.
*/

#[cfg(feature = "arrow")]
pub(crate) mod arrow_opt {
    use crate::table::slice_impl::SliceImpl;
    use arrow::buffer::NullBuffer;
    use num_traits::Float;
    use std::hash::Hash;

    pub(crate) fn parallel_approx_string_slice<'a, S: SliceImpl<&'a str> + Send + Sync>(
        slice: &S,
        bin_edges: &super::CategoricalBinEdges<String>,
        thread_count: usize,
    ) -> (Vec<f64>, f64) {
        let n_bins = bin_edges.n_bins();

        std::thread::scope(|s| {
            slice
                .chunk_indexes(thread_count)
                .map(|index_range| {
                    s.spawn(|| {
                        let mut null_count = 0_f64;
                        let mut hist = vec![0_f64; n_bins];

                        for idx in index_range {
                            if let Some(item) = slice.get(idx) {
                                hist[bin_edges.resolve_bin(item)] += 1_f64;
                            } else {
                                null_count += 1_f64;
                            }
                        }
                        (hist, null_count)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|t| t.join().unwrap())
                .fold(
                    (vec![0_f64; n_bins], 0_f64),
                    |(mut acc_hist, mut acc_null_count), (local_hist, local_null_count)| {
                        acc_null_count += local_null_count;
                        acc_hist
                            .iter_mut()
                            .zip(local_hist.iter())
                            .for_each(|(a, b)| *a += b);
                        (acc_hist, acc_null_count)
                    },
                )
        })
    }

    pub(crate) fn parallel_approx_boolean_slice<'a, S: SliceImpl<bool> + Send + Sync>(
        slice: &S,
        bin_edges: &super::CategoricalBinEdges<bool>,
        thread_count: usize,
    ) -> (Vec<f64>, f64) {
        let n_bins = bin_edges.n_bins();

        std::thread::scope(|s| {
            slice
                .chunk_indexes(thread_count)
                .map(|index_range| {
                    s.spawn(|| {
                        let mut null_count = 0_f64;
                        let mut hist = vec![0_f64; n_bins];

                        for idx in index_range {
                            if let Some(item) = slice.get(idx) {
                                hist[bin_edges.resolve_bin(&item)] += 1_f64;
                            } else {
                                null_count += 1_f64;
                            }
                        }
                        (hist, null_count)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|t| t.join().unwrap())
                .fold(
                    (vec![0_f64; n_bins], 0_f64),
                    |(mut acc_hist, mut acc_null_count), (local_hist, local_null_count)| {
                        acc_null_count += local_null_count;
                        acc_hist
                            .iter_mut()
                            .zip(local_hist.iter())
                            .for_each(|(a, b)| *a += b);
                        (acc_hist, acc_null_count)
                    },
                )
        })
    }

    pub(crate) fn parallel_approx_arrow_cont<T: Float + Send + Sync>(
        dataset: &[T],
        bin_edges: &super::ContinuousBinEdges<T>,
        null_buffer: Option<&NullBuffer>,
        thread_count: usize,
    ) -> (Vec<f64>, f64) {
        if let Some(null_buff) = null_buffer {
            cont_inner(dataset, bin_edges, null_buff, thread_count)
        } else {
            (
                super::continuous::parallel_approx_dataset(dataset, bin_edges, thread_count),
                0_f64,
            )
        }
    }

    fn cont_inner<T: Float + Send + Sync>(
        dataset: &[T],
        bin_edges: &super::ContinuousBinEdges<T>,
        null_buffer: &NullBuffer,
        thread_count: usize,
    ) -> (Vec<f64>, f64) {
        let n = dataset.len();
        let chunk_size = (n + thread_count - 1) / thread_count;
        let n_bins = bin_edges.n_bins();

        std::thread::scope(|s| {
            dataset
                .chunks(chunk_size) // last slice is remainder
                .enumerate()
                .map(|(shard_offset, dataset_chunk)| {
                    s.spawn(move || {
                        let mut null_count = 0_f64;
                        let mut local_hist = vec![0_f64; n_bins];
                        let shard_base_offset = shard_offset * chunk_size;
                        for (local_offset, ex) in dataset_chunk.iter().enumerate() {
                            let global_offset = shard_base_offset + local_offset;
                            if null_buffer.is_valid(global_offset) {
                                local_hist[bin_edges.resolve_bin(*ex)] += 1_f64;
                            } else {
                                null_count += 1_f64;
                            }
                        }
                        (local_hist, null_count)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|t| t.join().unwrap()) // safe unwrap: thread will not panic
                .fold(
                    (vec![0_f64; n_bins], 0_f64),
                    |(mut acc_hist, mut acc_null_count), (local_hist, local_null_count)| {
                        acc_null_count += local_null_count;
                        acc_hist
                            .iter_mut()
                            .zip(local_hist.iter())
                            .for_each(|(a, b)| *a += b);
                        (acc_hist, acc_null_count)
                    },
                )
        })
    }

    pub(crate) fn parallel_approx_arrow_cat<T: Hash + Ord + Clone + Send + Sync>(
        dataset: &[T],
        bin_edges: &super::CategoricalBinEdges<T>,
        null_buffer: Option<&NullBuffer>,
        thread_count: usize,
    ) -> (Vec<f64>, f64) {
        if let Some(null_buff) = null_buffer {
            cat_inner(dataset, bin_edges, null_buff, thread_count)
        } else {
            (
                super::categorical::parallel_approx_dataset(dataset, bin_edges),
                0_f64,
            )
        }
    }

    fn cat_inner<T: Hash + Ord + Clone + Send + Sync>(
        dataset: &[T],
        bin_edges: &super::CategoricalBinEdges<T>,
        null_buffer: &NullBuffer,
        thread_count: usize,
    ) -> (Vec<f64>, f64) {
        let n = dataset.len();
        let chunk_size = (n + thread_count - 1) / thread_count;
        let n_bins = bin_edges.n_bins();

        std::thread::scope(|s| {
            dataset
                .chunks(chunk_size) // last slice is remainder
                .enumerate()
                .map(|(shard_offset, dataset_chunk)| {
                    s.spawn(move || {
                        let mut null_count = 0_f64;
                        let mut local_hist = vec![0_f64; n_bins];
                        let shard_base_offset = shard_offset * chunk_size;
                        for (local_offset, ex) in dataset_chunk.iter().enumerate() {
                            let global_offset = shard_base_offset + local_offset;
                            if null_buffer.is_valid(global_offset) {
                                local_hist[bin_edges.resolve_bin(ex)] += 1_f64;
                            } else {
                                null_count += 1_f64;
                            }
                        }
                        (local_hist, null_count)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|t| t.join().unwrap()) // safe unwrap: thread will not panic
                .fold(
                    (vec![0_f64; n_bins], 0_f64),
                    |(mut acc_hist, mut acc_null_count), (local_hist, local_null_count)| {
                        acc_null_count += local_null_count;
                        acc_hist
                            .iter_mut()
                            .zip(local_hist.iter())
                            .for_each(|(a, b)| *a += b);
                        (acc_hist, acc_null_count)
                    },
                )
        })
    }
}

pub(crate) mod continuous {
    use num_traits::Float;

    pub(crate) fn parallel_approx_dataset<T: Float + Send + Sync>(
        dataset: &[T],
        bin_edges: &super::ContinuousBinEdges<T>,
        thread_count: usize,
    ) -> Vec<f64> {
        let n = dataset.len();
        let chunk_size = (n + thread_count - 1) / thread_count;
        let n_bins = bin_edges.n_bins();

        let hist = std::thread::scope(|s| {
            dataset
                .chunks(chunk_size) // last slice is remainder
                .map(|dataset_chunk| {
                    s.spawn(|| {
                        let mut local_hist = vec![0_f64; n_bins];
                        for ex in dataset_chunk.iter() {
                            local_hist[bin_edges.resolve_bin(*ex)] += 1_f64;
                        }
                        local_hist
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|t| t.join().unwrap()) // safe unwrap: thread will not panic
                .fold(vec![0_f64; n_bins], |mut acc, local| {
                    acc.iter_mut().zip(local.iter()).for_each(|(a, b)| *a += b);
                    acc
                })
        });
        hist
    }

    pub(crate) fn parallel_approx_dataset_nullable<T: Float + Send + Sync>(
        dataset: &[Option<T>],
        bin_edges: &super::ContinuousBinEdges<T>,
        thread_count: usize,
    ) -> (Vec<f64>, f64) {
        let n = dataset.len();
        let chunk_size = (n + thread_count - 1) / thread_count;
        let n_bins = bin_edges.n_bins();

        let hist = std::thread::scope(|s| {
            dataset
                .chunks(chunk_size) // last slice is remainder
                .map(|dataset_chunk| {
                    s.spawn(|| {
                        let mut count_none = 0_f64;
                        let mut local_hist = vec![0_f64; n_bins];
                        for example in dataset_chunk.iter() {
                            match example {
                                Some(ex) => local_hist[bin_edges.resolve_bin(*ex)] += 1_f64,
                                None => count_none += 1_f64,
                            }
                        }
                        (local_hist, count_none)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|t| t.join().unwrap()) // safe unwrap: thread will not panic
                .fold(
                    (vec![0_f64; n_bins], 0_f64),
                    |(mut acc, mut none_c_acc), (local, null_count)| {
                        none_c_acc += null_count;
                        acc.iter_mut().zip(local.iter()).for_each(|(a, b)| *a += b);
                        (acc, none_c_acc)
                    },
                )
        });
        hist
    }
}

pub(crate) mod categorical {
    use std::hash::Hash;

    pub(crate) fn parallel_approx_dataset<T: Hash + Ord + Clone + Send + Sync>(
        dataset: &[T],
        baseline: &super::CategoricalBinEdges<T>,
    ) -> Vec<f64> {
        let n = dataset.len();
        let thread_count = super::get_thread_count(n);
        let chunk_size = (n + thread_count - 1) / thread_count;
        let n_bins = baseline.n_bins();

        let hist = std::thread::scope(|s| {
            dataset
                .chunks(chunk_size)
                .map(|dataset_chunk| {
                    s.spawn(|| {
                        let mut local_hist = vec![0_f64; n_bins];
                        for ex in dataset_chunk.iter() {
                            local_hist[baseline.resolve_bin(ex)] += 1_f64;
                        }
                        local_hist
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|t| t.join().unwrap()) // safe unwrap: thread will not panic
                .fold(vec![0_f64; n_bins], |mut acc, local| {
                    acc.iter_mut().zip(local.iter()).for_each(|(a, b)| *a += b);
                    acc
                })
        });
        hist
    }

    pub(crate) fn parallel_approx_dataset_nullable<T: Hash + Ord + Clone + Send + Sync>(
        dataset: &[Option<T>],
        bin_edges: &super::CategoricalBinEdges<T>,
        thread_count: usize,
    ) -> (Vec<f64>, f64) {
        let n = dataset.len();
        let chunk_size = (n + thread_count - 1) / thread_count;
        let n_bins = bin_edges.n_bins();

        let hist = std::thread::scope(|s| {
            dataset
                .chunks(chunk_size) // last slice is remainder
                .map(|dataset_chunk| {
                    s.spawn(|| {
                        let mut count_none = 0_f64;
                        let mut local_hist = vec![0_f64; n_bins];
                        for example in dataset_chunk.iter() {
                            match example {
                                Some(e) => local_hist[bin_edges.resolve_bin(e)] += 1_f64,
                                None => count_none += 1_f64,
                            }
                        }
                        (local_hist, count_none)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|t| t.join().unwrap()) // safe unwrap: thread will not panic
                .fold(
                    (vec![0_f64; n_bins], 0_f64),
                    |(mut acc, mut none_c_acc), (local, null_count)| {
                        none_c_acc += null_count;
                        acc.iter_mut().zip(local.iter()).for_each(|(a, b)| *a += b);
                        (acc, none_c_acc)
                    },
                )
        });
        hist
    }
}
