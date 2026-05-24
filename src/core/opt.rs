use super::{CategoricalBinEdges, ContinuousBinEdges, NullableCategoricalBinEdges};
use crate::constants::get_thread_count;

/*
* Methods to approximate the distribution histogram across threads.
* Derive an approximate number of threads.
* Compute thread local distribution.
* Reduce local element wise count into global bins by folding the local bins.
*/

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
        bin_edges: &super::NullableCategoricalBinEdges<T>,
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
                            match bin_edges.resolve_bin(example) {
                                Some(bin) => local_hist[bin] += 1_f64,
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
