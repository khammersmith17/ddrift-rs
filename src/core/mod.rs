pub mod bin_edges;
pub mod dataset_view;
#[cfg(feature = "arrow")]
pub(crate) mod ddrift_arrow;
pub mod distribution;
pub mod drift_metrics;
pub mod error;
use bin_edges::{CategoricalBinEdges, ContinuousBinEdges};
mod opt;
use crate::constants::get_thread_count;
use ahash::{HashMap, HashMapExt};
use error::DriftError;
use num_traits::Float;
use std::collections::BTreeMap;
use std::hash::Hash;

pub(crate) fn compute_dataset_from_bins_continuous<T: Float + Send + Sync>(
    dataset: &[T],
    edges: &ContinuousBinEdges<T>,
) -> Vec<f64> {
    let thread_count = get_thread_count(dataset.len());
    if thread_count > 1 {
        opt::continuous::parallel_approx_dataset(dataset, edges, thread_count)
    } else {
        compute_dataset_from_bins_continuous_seq(dataset, edges)
    }
}

pub(crate) fn compute_nullable_dataset_from_bins_continuous<T: Float + Send + Sync>(
    dataset: &[Option<T>],
    edges: &ContinuousBinEdges<T>,
) -> (Vec<f64>, f64) {
    let thread_count = get_thread_count(dataset.len());
    if thread_count > 1 {
        opt::continuous::parallel_approx_dataset_nullable(dataset, edges, thread_count)
    } else {
        compute_nullable_dataset_from_bins_continuous_seq(dataset, edges)
    }
}

fn compute_dataset_from_bins_continuous_seq<T: Float>(
    dataset: &[T],
    edges: &ContinuousBinEdges<T>,
) -> Vec<f64> {
    let mut hist = vec![0_f64; edges.n_bins()];
    dataset
        .iter()
        .for_each(|e| hist[edges.resolve_bin(*e)] += 1_f64);
    hist
}

pub(crate) fn compute_dataset_from_bins_categorical_parallel<
    'a,
    T: Hash + Ord + Clone + Send + Sync,
>(
    dataset: &'a [T],
    edges: &'a CategoricalBinEdges<T>,
) -> Vec<f64> {
    let thread_count = get_thread_count(dataset.len());
    if thread_count > 1 {
        opt::categorical::parallel_approx_dataset(dataset, edges)
    } else {
        compute_dataset_from_bins_categorical(dataset, edges)
    }
}

fn compute_nullable_dataset_from_bins_continuous_seq<T: Float>(
    dataset: &[Option<T>],
    edges: &ContinuousBinEdges<T>,
) -> (Vec<f64>, f64) {
    let mut null_count = 0_f64;
    let mut hist = vec![0_f64; edges.n_bins()];
    dataset.iter().for_each(|e_opt| {
        if let Some(e) = e_opt {
            hist[edges.resolve_bin(*e)] += 1_f64
        } else {
            null_count += 1_f64;
        }
    });
    (hist, null_count)
}

pub(crate) fn compute_dataset_from_nullable_bins_categorical<'a, T: Hash + Ord + Clone>(
    dataset: &'a [Option<T>],
    edges: &'a CategoricalBinEdges<T>,
) -> (Vec<f64>, f64) {
    let mut hist = vec![0_f64; edges.n_bins()];
    let mut null_n = 0_f64;
    dataset.iter().for_each(|e_opt| {
        if let Some(e) = e_opt {
            hist[edges.resolve_bin(e)] += 1_f64
        } else {
            null_n += 1_f64
        }
    });
    (hist, null_n)
}

pub(crate) fn compute_dataset_from_nullable_bins_categorical_parallel<
    'a,
    T: Hash + Ord + Clone + Send + Sync,
>(
    dataset: &'a [Option<T>],
    edges: &'a CategoricalBinEdges<T>,
) -> (Vec<f64>, f64) {
    let thread_count = get_thread_count(dataset.len());
    if thread_count > 1 {
        opt::categorical::parallel_approx_dataset_nullable(dataset, edges, thread_count)
    } else {
        compute_dataset_from_nullable_bins_categorical(dataset, edges)
    }
}

pub(crate) fn compute_dataset_from_bins_categorical<'a, T: Hash + Ord + Clone>(
    dataset: &'a [T],
    edges: &'a CategoricalBinEdges<T>,
) -> Vec<f64> {
    let mut hist = vec![0_f64; edges.n_bins()];
    dataset
        .iter()
        .for_each(|e| hist[edges.resolve_bin(e)] += 1_f64);
    hist
}

/// Defines the lookup map for nullable categorical fields, and constructs the baseline histogram for drift
/// at "runtime".
pub(crate) fn nullable_categorical_derive_baseline_state<T: Hash + Ord + Clone>(
    baseline_dataset: &[Option<T>],
) -> Result<(HashMap<T, usize>, Vec<f64>, usize), DriftError> {
    if baseline_dataset.is_empty() {
        return Err(DriftError::EmptyBaselineData);
    }

    let total_n = baseline_dataset.len() as f64;
    let mut null_n = 0_usize;

    let mut initial_bins: BTreeMap<T, f64> = BTreeMap::new();
    for cat in baseline_dataset.iter() {
        let Some(example) = cat else {
            null_n += 1;
            continue;
        };

        if let Some(count) = initial_bins.get_mut(example) {
            *count += 1_f64;
        } else {
            initial_bins.insert(example.clone(), 1_f64);
        }
    }

    // Preallocate space for cardinatity of the dataset + 1
    // The additional bin is reserved for data values not observed in the baseline dataset
    let mut baseline_bins = vec![0_f64; initial_bins.len() + 1_usize];
    let mut idx_map: HashMap<T, usize> = HashMap::with_capacity(initial_bins.len());

    let nonnull_n = total_n - null_n as f64;

    for (i, (key, count)) in initial_bins.into_iter().enumerate() {
        idx_map.insert(key, i);
        baseline_bins[i] = count / nonnull_n;
    }
    Ok((idx_map, baseline_bins, null_n))
}

/// Defines the lookup map for categorical fields, and constructs the baseline histogram for drift
/// at "runtime".
pub(crate) fn categorical_derive_baseline_state<T: Hash + Ord + Clone>(
    baseline_dataset: &[T],
) -> Result<(HashMap<T, usize>, Vec<f64>), DriftError> {
    if baseline_dataset.is_empty() {
        return Err(DriftError::EmptyBaselineData);
    }

    // Count occurance of each value in the dataset.
    // BTreeMap is used to have deterministic ordering of bins.
    let initial_bins: BTreeMap<T, f64> =
        baseline_dataset
            .iter()
            .fold(BTreeMap::new(), |mut bin_acc, example| {
                if let Some(c) = bin_acc.get_mut(example) {
                    *c += 1_f64;
                } else {
                    bin_acc.insert(example.clone(), 1_f64);
                }
                bin_acc
            });

    // Preallocate space for cardinatity of the dataset + 1.
    // The additional bin is reserved for data values not observed in the baseline dataset
    let mut baseline_bins = vec![0_f64; initial_bins.len() + 1_usize];
    let mut idx_map: HashMap<T, usize> = HashMap::with_capacity(initial_bins.len());

    for (i, (key, count)) in initial_bins.into_iter().enumerate() {
        idx_map.insert(key, i);
        baseline_bins[i] = count;
    }
    Ok((idx_map, baseline_bins))
}
