use crate::{
    core::{
        categorical_derive_baseline_state,
        error::{DriftError, DriftExportError},
        nullable_categorical_derive_baseline_state,
    },
    export::CategoricalDriftBaselineExport,
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
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BaselineCategoricalBins<T: Hash + Ord + Clone> {
    pub(crate) idx_map: HashMap<T, usize>,
    pub(crate) baseline_bins: Vec<f64>,
    pub(crate) n: f64,
}

impl<T: Hash + Ord + Clone + serde::Serialize> TryInto<CategoricalDriftBaselineExport>
    for BaselineCategoricalBins<T>
{
    type Error = serde_json::Error;
    fn try_into(self) -> Result<CategoricalDriftBaselineExport, Self::Error> {
        let BaselineCategoricalBins {
            idx_map,
            baseline_bins: baseline_hist,
            n,
        } = self;

        let value_set: BTreeSet<T> = idx_map.into_iter().map(|(key, _)| key).collect();
        let mut baseline_values: Vec<serde_json::Value> = Vec::with_capacity(value_set.len());
        for value in value_set.into_iter() {
            baseline_values.push(serde_json::to_value(value)?);
        }

        Ok(CategoricalDriftBaselineExport {
            baseline_hist,
            baseline_values,
            n,
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
            n,
        } = export;

        if baseline_hist.len() - 1 != baseline_values.len() {
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

        Ok(BaselineCategoricalBins {
            baseline_bins: baseline_hist,
            idx_map,
            n,
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
        Ok(BaselineCategoricalBins {
            idx_map,
            baseline_bins,
            n: baseline_data.len() as f64,
        })
    }

    /// Resolve the bin idx for a particular key, otherwise return out the bin reserved for the
    /// "other" bucket.
    pub(crate) fn resolve_bin<Q>(&self, key: &Q) -> usize
    where
        T: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if let Some(idx) = self.idx_map.get(key) {
            *idx
        } else {
            self.baseline_bins.len() - 1
        }
    }

    /// Export the baseline histogram.
    pub(crate) fn export_baseline(&self) -> HashMap<T, f64> {
        self.idx_map
            .iter()
            .map(|(feat_name, i)| (feat_name.clone(), self.baseline_bins[*i]))
            .collect()
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

#[derive(Clone, Debug, PartialEq)]
pub struct NullableBaselineCategoricalBins<T: Hash + Ord + Clone> {
    pub(crate) idx_map: HashMap<T, usize>,
    pub(crate) baseline_bins: Vec<f64>,
    pub(crate) total_n: f64,
    pub(crate) null_n: f64,
}

impl<T: Hash + Ord + Clone> NullableBaselineCategoricalBins<T> {
    pub(crate) fn new(
        baseline_data: &[Option<T>],
    ) -> Result<NullableBaselineCategoricalBins<T>, DriftError> {
        let (idx_map, baseline_bins, null_count) =
            nullable_categorical_derive_baseline_state(baseline_data)?;

        Ok(NullableBaselineCategoricalBins {
            idx_map,
            baseline_bins,
            total_n: baseline_data.len() as f64,
            null_n: null_count as f64,
        })
    }

    pub(crate) fn get_baseline_hist(&self) -> &[f64] {
        &self.baseline_bins
    }

    /// Resolve the bin idx for a particular key, otherwise return out the bin reserved for the
    /// "other" bucket.
    pub(crate) fn resolve_bin<Q>(&self, key_opt: &Option<Q>) -> Option<usize>
    where
        T: Borrow<Q>,
        Q: Hash + Eq,
    {
        let Some(key) = key_opt else {
            return None;
        };

        if let Some(idx) = self.idx_map.get(key) {
            Some(*idx)
        } else {
            Some(self.baseline_bins.len() - 1)
        }
    }

    /// Export the baseline histogram.
    pub(crate) fn export_baseline(&self) -> HashMap<T, f64> {
        self.idx_map
            .iter()
            .map(|(feat_name, i)| (feat_name.clone(), self.baseline_bins[*i]))
            .collect()
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
