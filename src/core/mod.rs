pub mod bin_edges;
pub mod distribution;
pub mod drift_metrics;
pub mod error;
use bin_edges::{CategoricalBinEdges, ContinuousBinEdges, NullableCategoricalBinEdges};
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

pub(crate) fn compute_dataset_from_bins_continuous_null_parallel<T: Float + Send + Sync>(
    dataset: &[Option<T>],
    edges: &ContinuousBinEdges<T>,
) -> (Vec<f64>, usize) {
    opt::continuous::parallel_approx_dataset_nullable(
        dataset,
        edges,
        get_thread_count(dataset.len()),
    )
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

pub(crate) fn compute_dataset_from_bins_categorical_parallel<'a, T: Hash + Ord + Clone + Sync>(
    dataset: &'a [T],
    edges: &'a CategoricalBinEdges<T>,
) -> Vec<f64> {
    opt::categorical::parallel_approx_dataset(dataset, edges)
}

pub(crate) fn compute_dataset_from_nullable_bins_categorical<'a, T: Hash + Ord + Clone>(
    dataset: &'a [Option<T>],
    edges: &'a NullableCategoricalBinEdges<T>,
) -> (Vec<f64>, f64) {
    let mut hist = vec![0_f64; edges.n_bins()];
    let mut null_n = 0_f64;
    dataset.iter().for_each(|e| {
        if let Some(idx) = edges.resolve_bin(e) {
            hist[idx] += 1_f64
        } else {
            null_n += 1_f64
        }
    });
    (hist, null_n)
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

// Take the baseline bin counts and compute the proportional bin sizes based on total population
// size.
#[inline]
pub(crate) fn compute_new_hist_prob(
    num_items: usize,
    hist: &[f64],
) -> Result<Vec<f64>, DriftError> {
    let total_n = num_items as f64;
    if total_n == 0_f64 {
        return Err(DriftError::EmptyRuntimeData);
    }
    let bl_hist = hist.iter().map(|n| *n / total_n).collect::<Vec<f64>>();
    Ok(bl_hist)
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
    let n = baseline_dataset.len() as f64;

    let mut initial_bins: BTreeMap<T, f64> = BTreeMap::new();
    for cat in baseline_dataset.iter() {
        if let Some(count) = initial_bins.get_mut(cat) {
            *count += 1_f64;
        } else {
            initial_bins.insert(cat.clone(), 1_f64);
        }
    }

    // Preallocate space for cardinatity of the dataset + 1
    // The additional bin is reserved for data values not observed in the baseline dataset
    let mut baseline_bins = vec![0_f64; initial_bins.len() + 1_usize];
    let mut idx_map: HashMap<T, usize> = HashMap::with_capacity(initial_bins.len());

    for (i, (key, count)) in initial_bins.into_iter().enumerate() {
        idx_map.insert(key, i);
        baseline_bins[i] = count / n;
    }
    Ok((idx_map, baseline_bins))
}

#[cfg(test)]
mod core_drift_test {
    use super::*;
    #[test]
    fn test_new_hist_prob() {
        let bl_hist = vec![10.0, 20.0, 30.0, 40.0];
        let base: Vec<f64> = vec![0.10, 0.20, 0.30, 0.40];
        let test_bins = compute_new_hist_prob(100, &bl_hist).unwrap();
        assert_eq!(base, test_bins);
    }
}
