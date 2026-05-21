use crate::{
    core::{
        bin_edges::{CategoricalBinEdges, NullableCategoricalBinEdges},
        categorical_derive_baseline_state,
        error::{DriftError, DriftExportError},
        nullable_categorical_derive_baseline_state,
    },
    export::{CategoricalDriftBaselineExport, NullableCategoricalDriftBaselineExport},
};
use ahash::HashMap;
use std::borrow::Borrow;
use std::collections::BTreeSet;
use std::hash::Hash;

/*
* Trait bounds here enforce that the categorical values must be hashable to be stored as keys in
* the lookup app, comparable, and
* */

// idx_map holds the bin for a particular data value.
// Baseline bins are the histogram generated on baseline data, and other label represents the
// "other" bucket for when a discrete value not seen in the baseline set is observed.
#[derive(Clone, Debug)]
pub struct BaselineCategoricalBins<T: Hash + Ord + Clone> {
    pub(crate) bin_edges: CategoricalBinEdges<T>,
    pub(crate) baseline_bins: Vec<f64>,
    pub(crate) sample_size: f64,
}

impl<T: Hash + Ord + Clone + serde::Serialize> TryFrom<BaselineCategoricalBins<T>>
    for CategoricalDriftBaselineExport
{
    type Error = serde_json::Error;
    fn try_from(baseline: BaselineCategoricalBins<T>) -> Result<Self, Self::Error> {
        let BaselineCategoricalBins {
            bin_edges,
            baseline_bins: baseline_hist,
            sample_size,
        } = baseline;

        let value_set: BTreeSet<T> = bin_edges.0.into_iter().map(|(key, _)| key).collect();
        let mut baseline_values: Vec<serde_json::Value> = Vec::with_capacity(value_set.len());
        for value in value_set.into_iter() {
            baseline_values.push(serde_json::to_value(value)?);
        }

        Ok(CategoricalDriftBaselineExport {
            baseline_hist,
            baseline_values,
            sample_size,
        })
    }
}

impl<T: Hash + Ord + Clone + serde::de::DeserializeOwned> TryFrom<CategoricalDriftBaselineExport>
    for BaselineCategoricalBins<T>
{
    type Error = DriftExportError;
    fn try_from(export: CategoricalDriftBaselineExport) -> Result<Self, Self::Error> {
        let CategoricalDriftBaselineExport {
            baseline_hist,
            baseline_values,
            sample_size,
        } = export;

        if baseline_hist.is_empty() || baseline_hist.len() - 1 != baseline_values.len() {
            return Err(DriftExportError::InvalidDataShape);
        }
        let mut labels: BTreeSet<T> = BTreeSet::new();

        for v in baseline_values.into_iter() {
            labels.insert(serde_json::from_value(v)?);
        }

        let idx_map: HashMap<T, usize> = labels
            .into_iter()
            .enumerate()
            .map(|(i, label)| (label, i))
            .collect();

        let bin_edges = CategoricalBinEdges::new(idx_map);

        Ok(BaselineCategoricalBins {
            baseline_bins: baseline_hist,
            bin_edges,
            sample_size,
        })
    }
}

impl<T: Hash + Ord + Clone + serde::de::DeserializeOwned> BaselineCategoricalBins<T> {
    pub(crate) fn new_from_export(
        export: CategoricalDriftBaselineExport,
    ) -> Result<BaselineCategoricalBins<T>, DriftExportError> {
        Self::try_from(export)
    }
}

/*
* Each value present in the baseline dataset is mapped to a bin in the histogram Vec.
* The furthest right, ie len(set(baseline data)) index in the histogram Vec is reserved for
* observed values that were not part of the baseline set
* */

impl<T: Hash + Ord + Clone> BaselineCategoricalBins<T> {
    // bins and index map, allocated bins, fill histogram with counts.
    pub(crate) fn new(baseline_data: &[T]) -> Result<BaselineCategoricalBins<T>, DriftError> {
        let (idx_map, baseline_bins) = categorical_derive_baseline_state(baseline_data)?;
        let bin_edges = CategoricalBinEdges::new(idx_map);
        Ok(BaselineCategoricalBins {
            bin_edges,
            baseline_bins,
            sample_size: baseline_data.len() as f64,
        })
    }

    pub(crate) fn bin_edges(&self) -> &CategoricalBinEdges<T> {
        &self.bin_edges
    }

    /// Resolve the bin idx for a particular key, otherwise return out the bin reserved for the
    /// "other" bucket.
    pub(crate) fn resolve_bin<Q>(&self, key: &Q) -> usize
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.bin_edges.resolve_bin(key)
    }

    pub(crate) fn population_size(&self) -> f64 {
        self.sample_size
    }

    pub(crate) fn n_bins(&self) -> usize {
        self.baseline_bins.len()
    }

    /// Redefine the baseline.
    pub(crate) fn reset(&mut self, baseline_data: &[T]) -> Result<(), DriftError> {
        *self = Self::new(baseline_data)?;
        Ok(())
    }
}

impl<T: Hash + Ord + Clone + serde::Serialize> TryFrom<NullableBaselineCategoricalBins<T>>
    for NullableCategoricalDriftBaselineExport
{
    type Error = serde_json::Error;
    fn try_from(baseline: NullableBaselineCategoricalBins<T>) -> Result<Self, Self::Error> {
        let NullableBaselineCategoricalBins {
            bin_edges,
            baseline_bins: baseline_hist,
            total_samples,
            null_samples,
        } = baseline;

        let value_set: BTreeSet<T> = bin_edges.0.into_iter().map(|(key, _)| key).collect();
        let mut baseline_values: Vec<serde_json::Value> = Vec::with_capacity(value_set.len());
        for value in value_set.into_iter() {
            baseline_values.push(serde_json::to_value(value)?);
        }

        Ok(NullableCategoricalDriftBaselineExport {
            baseline_hist,
            baseline_values,
            total_samples,
            null_samples,
        })
    }
}

#[derive(Clone, Debug)]
pub struct NullableBaselineCategoricalBins<T: Hash + Ord + Clone> {
    pub bin_edges: NullableCategoricalBinEdges<T>,
    pub(crate) baseline_bins: Vec<f64>,
    pub(crate) total_samples: f64,
    pub(crate) null_samples: f64,
}

impl<T: Hash + Ord + Clone> NullableBaselineCategoricalBins<T> {
    pub(crate) fn new(
        baseline_data: &[Option<T>],
    ) -> Result<NullableBaselineCategoricalBins<T>, DriftError> {
        let (idx_map, baseline_bins, null_count) =
            nullable_categorical_derive_baseline_state(baseline_data)?;

        let bin_edges = NullableCategoricalBinEdges::new(idx_map);

        Ok(NullableBaselineCategoricalBins {
            bin_edges,
            baseline_bins,
            total_samples: baseline_data.len() as f64,
            null_samples: null_count as f64,
        })
    }

    pub(crate) fn population_size(&self) -> f64 {
        self.total_samples - self.null_samples
    }

    pub(crate) fn get_baseline_hist(&self) -> &[f64] {
        &self.baseline_bins
    }

    pub(crate) fn bin_edges(&self) -> &NullableCategoricalBinEdges<T> {
        &self.bin_edges
    }

    /// Resolve the bin idx for a particular key, otherwise return out the bin reserved for the
    /// "other" bucket.
    pub(crate) fn resolve_bin<Q>(&self, key_opt: &Option<Q>) -> Option<usize>
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.bin_edges.resolve_bin(key_opt)
    }

    pub(crate) fn n_bins(&self) -> usize {
        self.baseline_bins.len()
    }

    /// Redefine the baseline.
    pub(crate) fn reset(&mut self, baseline_data: &[Option<T>]) -> Result<(), DriftError> {
        *self = Self::new(baseline_data)?;
        Ok(())
    }
}

impl<T: Hash + Ord + Clone + serde::de::DeserializeOwned>
    TryFrom<NullableCategoricalDriftBaselineExport> for NullableBaselineCategoricalBins<T>
{
    type Error = DriftExportError;
    fn try_from(export: NullableCategoricalDriftBaselineExport) -> Result<Self, Self::Error> {
        let NullableCategoricalDriftBaselineExport {
            baseline_hist,
            baseline_values,
            total_samples,
            null_samples,
        } = export;

        if baseline_hist.is_empty() || baseline_hist.len() - 1 != baseline_values.len() {
            return Err(DriftExportError::InvalidDataShape);
        }
        let mut labels: BTreeSet<T> = BTreeSet::new();

        for v in baseline_values.into_iter() {
            labels.insert(serde_json::from_value(v)?);
        }

        let idx_map: HashMap<T, usize> = labels
            .into_iter()
            .enumerate()
            .map(|(i, label)| (label, i))
            .collect();

        let bin_edges = NullableCategoricalBinEdges::new(idx_map);

        Ok(NullableBaselineCategoricalBins {
            baseline_bins: baseline_hist,
            bin_edges,
            total_samples,
            null_samples,
        })
    }
}

impl<T: Hash + Ord + Clone + serde::de::DeserializeOwned> NullableBaselineCategoricalBins<T> {
    pub(crate) fn new_from_export(
        export: NullableCategoricalDriftBaselineExport,
    ) -> Result<NullableBaselineCategoricalBins<T>, DriftExportError> {
        Self::try_from(export)
    }
}
