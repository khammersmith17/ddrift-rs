use crate::baseline::{
    BaselineCategoricalBins, BaselineContinuousBins, NullableBaselineCategoricalBins,
    NullableBaselineContinuousBins,
};
use crate::core::{bin_edges::CategoricalBinEdges, distribution::QuantileType, error::DriftError};
use ahash::HashMap;
use num_traits::Float;
use std::hash::Hash;

pub struct CategoricalDatasetView<T: Ord + Clone + Hash> {
    pub size: usize,
    pub bin_counts: HashMap<T, usize>,
}

impl<T: Ord + Clone + Hash> From<BaselineCategoricalBins<T>> for CategoricalDatasetView<T> {
    fn from(baseline: BaselineCategoricalBins<T>) -> CategoricalDatasetView<T> {
        let BaselineCategoricalBins {
            bin_edges,
            sample_size,
            baseline_bins,
        } = baseline;
        let CategoricalBinEdges(mut idx_map) = bin_edges;
        // For each key, the index in idx_map can be used to resolve the offset in the bin vector
        // of where the value count is stored.
        for (_key, idx) in idx_map.iter_mut() {
            let value_count = baseline_bins[*idx];
            *idx = value_count as usize;
        }
        CategoricalDatasetView {
            bin_counts: idx_map,
            size: sample_size as usize,
        }
    }
}

impl<T: Ord + Hash + Clone> CategoricalDatasetView<T> {
    pub fn new(dataset: &[T]) -> Result<CategoricalDatasetView<T>, DriftError> {
        let bl = BaselineCategoricalBins::new(dataset)?;
        Ok(bl.into())
    }
}

pub struct NullableCategoricalDatasetView<T: Ord + Clone + Hash> {
    pub size: usize,
    pub bin_counts: HashMap<T, usize>,
    pub null_count: usize,
}

impl<T: Ord + Clone + Hash> From<NullableBaselineCategoricalBins<T>>
    for NullableCategoricalDatasetView<T>
{
    fn from(baseline: NullableBaselineCategoricalBins<T>) -> NullableCategoricalDatasetView<T> {
        let NullableBaselineCategoricalBins {
            bin_edges,
            total_samples,
            null_samples,
            baseline_bins,
        } = baseline;
        let CategoricalBinEdges(mut idx_map) = bin_edges;
        // For each key, the index in idx_map can be used to resolve the offset in the bin vector
        // of where the value count is stored.
        for (_key, idx) in idx_map.iter_mut() {
            let value_count = baseline_bins[*idx];
            *idx = value_count as usize;
        }
        NullableCategoricalDatasetView {
            bin_counts: idx_map,
            size: total_samples as usize,
            null_count: null_samples as usize,
        }
    }
}

impl<T: Ord + Hash + Clone> NullableCategoricalDatasetView<T> {
    pub fn new(dataset: &[Option<T>]) -> Result<NullableCategoricalDatasetView<T>, DriftError> {
        let bl = NullableBaselineCategoricalBins::new(dataset)?;
        Ok(bl.into())
    }
}

pub struct ContinuousDatasetView<T: Float> {
    pub quantile_bins: Vec<f64>,
    pub bin_edges: Vec<T>,
    pub size: usize,
}

impl<T: Float> From<BaselineContinuousBins<T>> for ContinuousDatasetView<T> {
    fn from(baseline: BaselineContinuousBins<T>) -> ContinuousDatasetView<T> {
        let BaselineContinuousBins {
            baseline_hist: quantile_bins,
            bin_edges: bin_edges_c,
            sample_size,
            ..
        } = baseline;
        ContinuousDatasetView {
            quantile_bins,
            bin_edges: bin_edges_c.bin_edges,
            size: sample_size as usize,
        }
    }
}

impl<T: Float + Send + Sync> ContinuousDatasetView<T> {
    pub fn new(
        dataset: &[T],
        quantile_type: Option<QuantileType>,
    ) -> Result<ContinuousDatasetView<T>, DriftError> {
        let bl = BaselineContinuousBins::new(dataset, quantile_type)?;
        Ok(bl.into())
    }
}

pub struct NullableContinuousDatasetView<T: Float> {
    pub quantile_bins: Vec<f64>,
    pub bin_edges: Vec<T>,
    pub size: usize,
    pub null_count: usize,
}

impl<T: Float> From<NullableBaselineContinuousBins<T>> for NullableContinuousDatasetView<T> {
    fn from(baseline: NullableBaselineContinuousBins<T>) -> NullableContinuousDatasetView<T> {
        let NullableBaselineContinuousBins {
            baseline_hist: quantile_bins,
            bin_edges: bin_edges_c,
            sample_size,
            null_count,
            ..
        } = baseline;
        NullableContinuousDatasetView {
            quantile_bins,
            bin_edges: bin_edges_c.bin_edges,
            size: sample_size as usize,
            null_count: null_count as usize,
        }
    }
}

impl<T: Float + Send + Sync> NullableContinuousDatasetView<T> {
    pub fn new(
        dataset: &[Option<T>],
        quantile_type: Option<QuantileType>,
    ) -> Result<NullableContinuousDatasetView<T>, DriftError> {
        let bl = NullableBaselineContinuousBins::new(dataset, quantile_type)?;
        Ok(bl.into())
    }
}
