use crate::{
    baseline::categorical::{BaselineCategoricalBins, NullableBaselineCategoricalBins},
    core::{
        compute_dataset_from_bins_categorical, compute_dataset_from_bins_categorical_parallel,
        compute_dataset_from_nullable_bins_categorical,
        compute_dataset_from_nullable_bins_categorical_parallel,
        drift_metrics::{
            CategoricalDriftMeasurement, DriftContainer, compute_drift_categorical,
            compute_drift_categorical_multi,
        },
        error::{DriftError, DriftExportError},
    },
    drift::{
        DriftComputation, DriftComputationMulti, NullableDriftComputation,
        NullableDriftComputationMulti,
    },
    export,
};
use std::hash::Hash;

pub struct NullableCategoricalDataDrift<T: Hash + Ord + Clone> {
    pub(crate) baseline: NullableBaselineCategoricalBins<T>,
    rt_bins: Vec<f64>,
    n: f64,
    null_n: f64,
}

impl<T: Hash + Ord + Clone> From<NullableBaselineCategoricalBins<T>>
    for NullableCategoricalDataDrift<T>
{
    fn from(baseline: NullableBaselineCategoricalBins<T>) -> NullableCategoricalDataDrift<T> {
        let rt_bins = vec![0_f64; baseline.n_bins()];
        NullableCategoricalDataDrift {
            baseline,
            rt_bins,
            n: 0_f64,
            null_n: 0_f64,
        }
    }
}

impl<T: Hash + Ord + Clone> DriftContainer for NullableCategoricalDataDrift<T> {
    fn baseline_bins(&self) -> &[f64] {
        self.baseline.get_baseline_hist()
    }

    fn runtime_bins(&self) -> &[f64] {
        &self.rt_bins
    }

    fn baseline_sample_size(&self) -> f64 {
        self.baseline.population_size()
    }

    fn runtime_sample_size(&self) -> f64 {
        self.n - self.null_n
    }
}

impl<T: Hash + Ord + Clone + serde::de::DeserializeOwned> NullableCategoricalDataDrift<T> {
    pub fn new_from_export(
        export: export::NullableCategoricalDriftBaselineExport,
    ) -> Result<NullableCategoricalDataDrift<T>, DriftExportError> {
        let baseline: NullableBaselineCategoricalBins<T> =
            NullableBaselineCategoricalBins::try_from(export)?;
        let rt_bins = vec![0_f64; baseline.n_bins()];
        Ok(NullableCategoricalDataDrift {
            baseline,
            rt_bins,
            n: 0_f64,
            null_n: 0_f64,
        })
    }
}

impl<T: Hash + Ord + Clone> NullableCategoricalDataDrift<T> {
    /// Construct a new instance from a baseline dataset. The baseline is used to build a
    /// label-frequency histogram with one bin per unique value, plus one reserved "other" bin
    /// for values not present in the baseline.
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if `baseline_data` is empty.
    pub fn new(baseline_data: &[Option<T>]) -> Result<NullableCategoricalDataDrift<T>, DriftError> {
        if baseline_data.is_empty() {
            return Err(DriftError::EmptyBaselineData);
        }

        let baseline = NullableBaselineCategoricalBins::new(baseline_data)?;
        let num_bins = baseline.baseline_bins.len();
        let rt_bins: Vec<f64> = vec![0_f64; num_bins];

        Ok(NullableCategoricalDataDrift {
            baseline,
            rt_bins,
            n: 0_f64,
            null_n: 0_f64,
        })
    }

    pub fn compute_drift(
        &mut self,
        runtime_data: &[Option<T>],
        drift_type: CategoricalDriftMeasurement,
    ) -> Result<NullableDriftComputation<CategoricalDriftMeasurement>, DriftError> {
        self.build_rt_hist(runtime_data)?;
        let null_percentage = self.null_n / self.n;
        let drift = compute_drift_categorical(self, drift_type);
        self.clear_rt();
        Ok(NullableDriftComputation {
            drift,
            null_percentage,
        })
    }

    pub fn compute_drift_multiple_criteria(
        &mut self,
        runtime_data: &[Option<T>],
        drift_types: &[CategoricalDriftMeasurement],
    ) -> Result<NullableDriftComputationMulti<CategoricalDriftMeasurement>, DriftError> {
        self.build_rt_hist(runtime_data)?;
        let null_percentage = self.null_n / self.n;
        let DriftComputationMulti { drift } = compute_drift_categorical_multi(self, drift_types);
        self.clear_rt();
        Ok(NullableDriftComputationMulti {
            drift: drift.into(),
            null_percentage,
        })
    }

    fn build_rt_hist(&mut self, runtime_data: &[Option<T>]) -> Result<(), DriftError> {
        if runtime_data.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        self.n = runtime_data.len() as f64;
        (self.rt_bins, self.null_n) =
            compute_dataset_from_nullable_bins_categorical(runtime_data, self.baseline.bin_edges());
        Ok(())
    }

    fn clear_rt(&mut self) {
        self.rt_bins.fill(0_f64);
        self.n = 0_f64;
        self.null_n = 0_f64;
    }
}

impl<T: Hash + Ord + Clone + Send + Sync> NullableCategoricalDataDrift<T> {
    pub fn compute_drift_par(
        &mut self,
        runtime_data: &[Option<T>],
        drift_type: CategoricalDriftMeasurement,
    ) -> Result<NullableDriftComputation<CategoricalDriftMeasurement>, DriftError> {
        self.build_rt_hist_par(runtime_data)?;
        let null_percentage = self.null_n / self.n;
        let drift = compute_drift_categorical(self, drift_type);
        self.clear_rt();
        Ok(NullableDriftComputation {
            drift,
            null_percentage,
        })
    }

    pub fn compute_drift_multiple_criteria_par(
        &mut self,
        runtime_data: &[Option<T>],
        drift_types: &[CategoricalDriftMeasurement],
    ) -> Result<NullableDriftComputationMulti<CategoricalDriftMeasurement>, DriftError> {
        self.build_rt_hist_par(runtime_data)?;
        let null_percentage = self.null_n / self.n;
        let DriftComputationMulti { drift } = compute_drift_categorical_multi(self, drift_types);
        self.clear_rt();
        Ok(NullableDriftComputationMulti {
            drift: drift.into(),
            null_percentage,
        })
    }

    fn build_rt_hist_par(&mut self, runtime_data: &[Option<T>]) -> Result<(), DriftError> {
        if runtime_data.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        self.n = runtime_data.len() as f64;
        (self.rt_bins, self.null_n) = compute_dataset_from_nullable_bins_categorical_parallel(
            runtime_data,
            self.baseline.bin_edges(),
        );
        Ok(())
    }
}

#[derive(Debug)]
pub struct CategoricalDataDrift<T: Hash + Ord + Clone> {
    pub(crate) baseline: BaselineCategoricalBins<T>,
    rt_bins: Vec<f64>,
    sample_size: f64,
}

impl<T: Hash + Ord + Clone> From<BaselineCategoricalBins<T>> for CategoricalDataDrift<T> {
    fn from(baseline: BaselineCategoricalBins<T>) -> CategoricalDataDrift<T> {
        let rt_bins = vec![0_f64; baseline.n_bins()];
        CategoricalDataDrift {
            baseline,
            rt_bins,
            sample_size: 0_f64,
        }
    }
}

impl<T: Hash + Ord + Clone> DriftContainer for CategoricalDataDrift<T> {
    fn baseline_bins(&self) -> &[f64] {
        &self.baseline.baseline_bins
    }

    fn runtime_bins(&self) -> &[f64] {
        &self.rt_bins
    }

    fn runtime_sample_size(&self) -> f64 {
        self.sample_size
    }

    fn baseline_sample_size(&self) -> f64 {
        self.baseline.population_size()
    }
}

impl<T: Hash + Ord + Clone + serde::de::DeserializeOwned> CategoricalDataDrift<T> {
    pub fn new_from_export(
        export: export::CategoricalDriftBaselineExport,
    ) -> Result<CategoricalDataDrift<T>, DriftExportError> {
        let baseline: BaselineCategoricalBins<T> = BaselineCategoricalBins::try_from(export)?;
        let rt_bins = vec![0_f64; baseline.n_bins()];
        Ok(CategoricalDataDrift {
            baseline,
            rt_bins,
            sample_size: 0_f64,
        })
    }
}

impl<T: Hash + Ord + Clone + Send + Sync> CategoricalDataDrift<T> {
    /// Construct a new instance from a baseline dataset. The baseline is used to build a
    /// label-frequency histogram with one bin per unique value, plus one reserved "other" bin
    /// for values not present in the baseline. This method requires a T that is Sync, thus
    /// the seperate method surface from the base method.
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if `baseline_data` is empty.
    pub fn new_par(baseline_data: &[T]) -> Result<CategoricalDataDrift<T>, DriftError> {
        if baseline_data.is_empty() {
            return Err(DriftError::EmptyBaselineData);
        }

        let baseline = BaselineCategoricalBins::new(baseline_data)?;
        let num_bins = baseline.baseline_bins.len();
        let rt_bins: Vec<f64> = vec![0_f64; num_bins];

        Ok(CategoricalDataDrift {
            baseline,
            rt_bins,
            sample_size: 0_f64,
        })
    }

    fn build_rt_hist_parallel(&mut self, runtime_data: &[T]) -> Result<(), DriftError> {
        if runtime_data.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        self.rt_bins =
            compute_dataset_from_bins_categorical_parallel(runtime_data, self.baseline.bin_edges());
        self.sample_size = runtime_data.len() as f64;
        Ok(())
    }

    /// Compute drift between the baseline and the provided runtime dataset. This method uses the
    /// internal implementation to compute the runtime dataset distribution, and thus requires T to
    /// be sync. If T is not Sync, then the base method can be used. The runtime data is
    /// binned against the baseline label map, drift is computed, and the runtime bins are
    /// cleared. Each call is stateless with respect to prior runtime data.
    ///
    /// To compute drift across multiple criteria, use [`CategoricalDataDrift::compute_drift_multiple_criteria`]
    ///
    /// Runtime labels not seen in the baseline are accumulated in the "other" bin.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if `runtime_data` is empty.
    pub fn compute_drift_par(
        &mut self,
        runtime_data: &[T],
        drift_type: CategoricalDriftMeasurement,
    ) -> Result<DriftComputation<CategoricalDriftMeasurement>, DriftError> {
        self.build_rt_hist_parallel(runtime_data)?;
        let drift = compute_drift_categorical(self, drift_type);
        self.clear_rt();
        Ok(drift)
    }

    /// Compute drift between the baseline and the provided runtime dataset for multiple drift
    /// metric types, leveraging the multiple threads to derive the runtime dataset distribution.
    /// The runtime data is binned against the baseline label map, drift is computed, and the
    /// runtime bins are cleared. Each call is stateless with respect to prior runtime data.
    ///
    /// This method is much more efficient for computing drift across multiple criteria as it only
    /// requires a single build of the runtime data distribution representation.
    ///
    /// Runtime labels not seen in the baseline are accumulated in the "other" bin.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if `runtime_data` is empty.
    pub fn compute_drift_multiple_criteria_par(
        &mut self,
        runtime_data: &[T],
        drift_types: &[CategoricalDriftMeasurement],
    ) -> Result<DriftComputationMulti<CategoricalDriftMeasurement>, DriftError> {
        self.build_rt_hist_parallel(runtime_data)?;
        let drift = compute_drift_categorical_multi(self, drift_types);
        self.clear_rt();
        Ok(drift)
    }
}

impl<T: Hash + Ord + Clone> CategoricalDataDrift<T> {
    /// Construct a new instance from a baseline dataset. The baseline is used to build a
    /// label-frequency histogram with one bin per unique value, plus one reserved "other" bin
    /// for values not present in the baseline.
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if `baseline_data` is empty.
    pub fn new(baseline_data: &[T]) -> Result<CategoricalDataDrift<T>, DriftError> {
        if baseline_data.is_empty() {
            return Err(DriftError::EmptyBaselineData);
        }

        let baseline = BaselineCategoricalBins::new(baseline_data)?;
        let num_bins = baseline.baseline_bins.len();
        let rt_bins: Vec<f64> = vec![0_f64; num_bins];

        Ok(CategoricalDataDrift {
            baseline,
            rt_bins,
            sample_size: 0_f64,
        })
    }

    /// Compute drift between the baseline and the provided runtime dataset. The runtime data is
    /// binned against the baseline label map, drift is computed, and the runtime bins are
    /// cleared. Each call is stateless with respect to prior runtime data.
    ///
    /// To compute drift across multiple criteria, use [`CategoricalDataDrift::compute_drift_multiple_criteria`]
    ///
    /// Runtime labels not seen in the baseline are accumulated in the "other" bin.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if `runtime_data` is empty.
    pub fn compute_drift(
        &mut self,
        runtime_data: &[T],
        drift_type: CategoricalDriftMeasurement,
    ) -> Result<DriftComputation<CategoricalDriftMeasurement>, DriftError> {
        self.build_rt_hist(runtime_data)?;
        let drift = compute_drift_categorical(self, drift_type);
        self.clear_rt();
        Ok(drift)
    }

    /// Compute drift between the baseline and the provided runtime dataset for multiple drift
    /// metric types. The runtime data is binned against the baseline label map, drift is computed,
    /// and the runtime bins are cleared. Each call is stateless with respect to prior runtime data.
    ///
    /// This method is much more efficient for computing drift across multiple criteria as it only
    /// requires a single build of the runtime data distribution representation.
    ///
    /// Runtime labels not seen in the baseline are accumulated in the "other" bin.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if `runtime_data` is empty.
    pub fn compute_drift_multiple_criteria(
        &mut self,
        runtime_data: &[T],
        drift_types: &[CategoricalDriftMeasurement],
    ) -> Result<DriftComputationMulti<CategoricalDriftMeasurement>, DriftError> {
        self.build_rt_hist(runtime_data)?;
        let drift = compute_drift_categorical_multi(self, drift_types);
        self.clear_rt();
        Ok(drift)
    }

    fn build_rt_hist(&mut self, runtime_data: &[T]) -> Result<(), DriftError> {
        if runtime_data.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        self.rt_bins =
            compute_dataset_from_bins_categorical(runtime_data, self.baseline.bin_edges());
        self.sample_size = runtime_data.len() as f64;
        Ok(())
    }

    fn clear_rt(&mut self) {
        self.rt_bins.fill(0_f64);
        self.sample_size = 0_f64;
    }

    /// Replace the baseline with a new dataset. The bin count is recomputed from the new data —
    /// the number of bins becomes the new cardinality plus one "other" bin. Any previously
    /// accumulated runtime bins are reset to zero.
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if `new_baseline` is empty.
    pub fn reset_baseline(&mut self, new_baseline: &[T]) -> Result<(), DriftError> {
        self.baseline.reset(new_baseline)?;
        let num_bins = self.baseline.baseline_bins.len();

        // pay the cost to reallocate bins in order to have correct size
        // not common path
        self.rt_bins = vec![0_f64; num_bins];
        Ok(())
    }

    pub fn num_bins(&self) -> usize {
        self.rt_bins.len()
    }
}

impl<T: Hash + Ord + Clone + serde::Serialize> CategoricalDataDrift<T> {
    pub fn export_baseline_state(
        self,
    ) -> Result<export::CategoricalDriftBaselineExport, serde_json::Error> {
        export::CategoricalDriftBaselineExport::try_from(self.baseline)
    }
}

impl<T: Hash + Ord + Clone + serde::Serialize> NullableCategoricalDataDrift<T> {
    pub fn export_baseline_state(
        self,
    ) -> Result<export::NullableCategoricalDriftBaselineExport, serde_json::Error> {
        export::NullableCategoricalDriftBaselineExport::try_from(self.baseline)
    }
}

#[cfg(test)]
mod categorical_test {
    use super::*;

    #[test]
    fn test_categorical_baseline_builds_expected_size() {
        let baseline = ["a", "b", "a", "c"];
        let psi = CategoricalDataDrift::new(&baseline).unwrap();

        // baseline has 3 real labels + OTHER bucket
        assert_eq!(psi.baseline.baseline_bins.len(), 4);
    }

    #[test]
    fn test_categorical_psi_zero_when_no_drift() {
        let baseline = ["a", "b", "a", "c"];
        let mut psi = CategoricalDataDrift::new(&baseline).unwrap();
        let runtime = ["a", "b", "a", "c"];
        let drift = psi
            .compute_drift(
                &runtime,
                CategoricalDriftMeasurement::PopulationStabilityIndex,
            )
            .unwrap();
        assert!(drift.drift_magnitude.abs() < 1e-9);
    }

    #[test]
    fn test_categorical_psi_detects_shift() {
        let baseline = ["a", "b", "a", "c"];
        let mut psi = CategoricalDataDrift::new(&baseline).unwrap();
        let runtime = ["x", "x", "x", "x"]; // go to other bucket
        let drift = psi
            .compute_drift(
                &runtime,
                CategoricalDriftMeasurement::PopulationStabilityIndex,
            )
            .unwrap();
        assert!(drift.drift_magnitude > 0.5);
    }

    #[test]
    fn categorical_batch_empty_baseline_returns_err() {
        let empty: &[&str] = &[];
        assert!(CategoricalDataDrift::new(empty).is_err());
    }

    #[test]
    fn categorical_batch_empty_runtime_returns_err() {
        let mut det = CategoricalDataDrift::new(&["a", "b", "c"]).unwrap();
        let empty: &[&str] = &[];
        assert!(
            det.compute_drift(empty, CategoricalDriftMeasurement::PopulationStabilityIndex)
                .is_err()
        );
    }

    #[test]
    fn categorical_batch_novel_label_routes_to_other_bin() {
        // runtime with only unseen labels should produce higher drift than runtime matching baseline
        let baseline = ["a", "b", "a", "b", "a"];
        let mut det = CategoricalDataDrift::new(&baseline).unwrap();

        let matching_drift = det
            .compute_drift(
                &["a", "b", "a", "b", "a"],
                CategoricalDriftMeasurement::PopulationStabilityIndex,
            )
            .unwrap();
        let novel_drift = det
            .compute_drift(
                &["x", "y", "z", "x", "y"],
                CategoricalDriftMeasurement::PopulationStabilityIndex,
            )
            .unwrap();

        assert!(novel_drift.drift_magnitude > matching_drift.drift_magnitude);
    }

    // --- all drift types ---

    #[test]
    fn categorical_batch_all_drift_types_finite_nonnegative() {
        let baseline = ["a", "a", "b", "b", "c"];
        let runtime = ["a", "b", "b", "c", "c"];
        let mut det = CategoricalDataDrift::new(&baseline).unwrap();

        for drift_type in [
            CategoricalDriftMeasurement::PopulationStabilityIndex,
            CategoricalDriftMeasurement::KullbackLeibler,
            CategoricalDriftMeasurement::JensenShannon,
            CategoricalDriftMeasurement::WassersteinDistance,
        ] {
            let v = det.compute_drift(&runtime, drift_type).unwrap();
            assert!(
                v.drift_magnitude.is_finite(),
                "{drift_type:?} produced non-finite value"
            );
            assert!(
                v.drift_magnitude >= 0.0,
                "{drift_type:?} produced negative value"
            );
        }
    }

    // --- reset_baseline ---

    #[test]
    fn categorical_batch_reset_baseline_changes_bin_count() {
        let mut det = CategoricalDataDrift::new(&["a", "b"]).unwrap();
        assert_eq!(det.rt_bins.len(), 3); // 2 labels + other

        det.reset_baseline(&["a", "b", "c", "d"]).unwrap();
        assert_eq!(det.rt_bins.len(), 5); // 4 labels + other
    }
}
