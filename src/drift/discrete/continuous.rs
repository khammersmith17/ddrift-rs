use crate::drift::{DriftComputation, NullableDriftComputation, NullableDriftComputationMulti};
use crate::{
    baseline::continuous::{BaselineContinuousBins, NullableBaselineContinuousBins},
    core::{
        compute_dataset_from_bins_continuous, compute_dataset_from_bins_continuous_null_parallel,
        distribution::QuantileType,
        drift_metrics::{
            ContinuousDriftType, DriftContainer, compute_drift_continuous,
            compute_drift_continuous_multi,
        },
        error::{DriftError, DriftExportError},
    },
    export,
};
use num_traits::Float;

/// Batch drift detector for continuous (floating-point) features. Compares a provided runtime
/// dataset against a fixed baseline distribution using histogram binning.
///
/// The baseline histogram is built once at construction and held until [`reset_baseline`] is
/// called. Runtime data is binned on each call to [`compute_drift`] and discarded immediately
/// after — no state is accumulated between calls.
///
/// # Bin count
///
/// The number of histogram bins is derived automatically from the baseline data using one of
/// three heuristics, selected via [`QuantileType`]:
///
/// - **[`FreedmanDiaconis`]** *(default)*: `width = 2 * IQR * n^(-1/3)`, `k = ceil((max - min) / width)`.
///   Robust to outliers. Preferred for most use cases.
/// - **[`Scott`]**: `width = 3.49 * σ * n^(-1/3)`. Assumes approximately normal data. Sensitive
///   to outliers in the tails.
/// - **[`Sturges`]**: `k = floor(ln(n)) + 1`. Simple log-based rule. Works best for small,
///   roughly normal datasets.
///
/// [`reset_baseline`]: ContinuousDataDrift::reset_baseline
/// [`compute_drift`]: ContinuousDataDrift::compute_drift
/// [`FreedmanDiaconis`]: QuantileType::FreedmanDiaconis
/// [`Scott`]: QuantileType::Scott
/// [`Sturges`]: QuantileType::Sturges
pub struct ContinuousDataDrift<T: Float> {
    baseline: BaselineContinuousBins<T>,
    rt_bins: Vec<f64>,
    sample_size: f64,
}

impl<T: Float> DriftContainer for ContinuousDataDrift<T> {
    fn baseline_bins(&self) -> &[f64] {
        &self.baseline.baseline_bins()
    }

    fn runtime_bins(&self) -> &[f64] {
        &self.rt_bins
    }

    fn baseline_sample_size(&self) -> f64 {
        self.baseline.population_size()
    }

    fn runtime_sample_size(&self) -> f64 {
        self.sample_size
    }
}

impl<T: Float + serde::de::DeserializeOwned> ContinuousDataDrift<T> {
    pub fn new_from_export(
        export: export::ContinuousDriftBaselineExport<T>,
    ) -> Result<ContinuousDataDrift<T>, DriftExportError> {
        let baseline = BaselineContinuousBins::new_from_export(export)?;
        let rt_bins = vec![0_f64; baseline.n_bins()];
        Ok(ContinuousDataDrift {
            baseline,
            rt_bins,
            sample_size: 0_f64,
        })
    }
}

impl<T: Float + Send + Sync> ContinuousDataDrift<T> {
    /// Construct a new instance from a baseline dataset. The baseline is sorted and used to
    /// define histogram bin edges.
    ///
    /// `quantile_type` controls how many bins are derived from the baseline. The bin count
    /// determines the resolution of the drift signal — more bins capture finer distributional
    /// shifts but require more runtime data per bin to be statistically meaningful. Pass `None`
    /// to use the default. Options:
    ///
    /// - [`FreedmanDiaconis`] *(default)*: `width = 2 * IQR * n^(-1/3)`. Robust to outliers.
    ///   Preferred for most use cases.
    /// - [`Scott`]: `width = 3.49 * σ * n^(-1/3)`. Assumes approximately normal data; sensitive
    ///   to outliers in the tails.
    /// - [`Sturges`]: `k = floor(log2(n)) + 1`. Simple rule that works best for small, roughly
    ///   normal datasets.
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if the baseline slice has fewer than 2 elements,
    /// or [`DriftError::NaNValueError`] if any value is NaN.
    ///
    /// [`FreedmanDiaconis`]: QuantileType::FreedmanDiaconis
    /// [`Scott`]: QuantileType::Scott
    /// [`Sturges`]: QuantileType::Sturges
    pub fn new_from_baseline(
        quantile_type: Option<QuantileType>,
        bl_slice: &[T],
    ) -> Result<ContinuousDataDrift<T>, DriftError> {
        let sample_size = bl_slice.len() as f64;
        let baseline = BaselineContinuousBins::new(bl_slice, quantile_type.unwrap_or_default())?;
        let rt_bins = vec![0_f64; baseline.baseline_bins().len()];
        Ok(ContinuousDataDrift {
            baseline,
            rt_bins,
            sample_size,
        })
    }

    /// Compute drift between the baseline and the provided runtime dataset. The runtime data is
    /// binned against the baseline histogram edges, drift is computed, and the runtime bins are
    /// cleared. Each call is stateless with respect to prior runtime data.
    ///
    /// To compute drift across multiple criteria, use [`ContinuousDataDrift::compute_drift_multiple_criteria`]
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if `runtime_data` is empty.
    pub fn compute_drift(
        &mut self,
        runtime_data: &[T],
        drift_type: ContinuousDriftType,
    ) -> Result<DriftComputation<ContinuousDriftType>, DriftError> {
        self.build_rt_hist(runtime_data)?;
        let drift = compute_drift_continuous(self, drift_type);
        self.clear_rt();
        Ok(drift)
    }

    /// Compute drift between the baseline and the provided runtime dataset for multiple data drift
    /// types. The runtime data is binned against the baseline histogram edges, drift is computed,
    /// and the runtime bins are cleared. Each call is stateless with respect to prior runtime data.
    ///
    /// This method is much more efficient for computing drift across multiple criteria as it only
    /// requires a single build of the runtime data distribution representation.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if `runtime_data` is empty.
    pub fn compute_drift_multiple_criteria(
        &mut self,
        runtime_data: &[T],
        drift_types: &[ContinuousDriftType],
    ) -> Result<Vec<DriftComputation<ContinuousDriftType>>, DriftError> {
        self.build_rt_hist(runtime_data)?;
        let drift = compute_drift_continuous_multi(self, drift_types);
        self.clear_rt();
        Ok(drift)
    }

    /// Replace the baseline with a new dataset. The bin count is recomputed from the new data
    /// using the same [`QuantileType`] as construction. Any previously accumulated runtime bins
    /// are reset to zero.
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if the slice has fewer than 2 elements.
    pub fn reset_baseline(&mut self, baseline_slice: &[T]) -> Result<(), DriftError> {
        self.baseline.reset(baseline_slice)?;
        self.init_runtime_containers();
        Ok(())
    }

    #[inline]
    fn build_rt_hist(&mut self, runtime_data: &[T]) -> Result<(), DriftError> {
        if runtime_data.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        self.rt_bins =
            compute_dataset_from_bins_continuous(runtime_data, &self.baseline.bin_edges());
        self.sample_size = runtime_data.len() as f64;
        Ok(())
    }

    fn init_runtime_containers(&mut self) {
        let len = self.baseline.baseline_bins().len();
        self.rt_bins = vec![0_f64; len];
    }

    fn clear_rt(&mut self) {
        self.rt_bins.fill(0.);
        self.sample_size = 0_f64;
    }
}

impl<T: Float> ContinuousDataDrift<T> {
    /// The number of histogram bins derived from the baseline dataset.
    pub fn n_bins(&self) -> usize {
        self.baseline.n_bins()
    }

    /// Export the baseline bin proportions. Each value represents the proportion of baseline
    /// samples that fell into the corresponding bin.
    pub fn export_baseline(&self) -> Vec<f64> {
        self.baseline.export_baseline()
    }
}

impl<T: Float + serde::Serialize> ContinuousDataDrift<T> {
    pub fn export_baseline_state(self) -> export::ContinuousDriftBaselineExport<T> {
        self.baseline.into()
    }
}

pub struct NullableContinuousDataDrift<T: Float> {
    baseline: NullableBaselineContinuousBins<T>,
    rt_bins: Vec<f64>,
    n: f64,
    null_n: f64,
}

impl<T: Float> DriftContainer for NullableContinuousDataDrift<T> {
    fn baseline_bins(&self) -> &[f64] {
        &self.baseline.baseline_bins()
    }

    fn runtime_bins(&self) -> &[f64] {
        &self.rt_bins
    }

    fn runtime_sample_size(&self) -> f64 {
        self.n - self.null_n
    }

    fn baseline_sample_size(&self) -> f64 {
        self.baseline.population_size()
    }
}

impl<T: Float + serde::de::DeserializeOwned> NullableContinuousDataDrift<T> {
    pub fn new_from_export(
        export: export::NullableContinuousDriftBaselineExport<T>,
    ) -> Result<NullableContinuousDataDrift<T>, DriftExportError> {
        let baseline = NullableBaselineContinuousBins::new_from_export(export)?;
        let rt_bins = vec![0_f64; baseline.n_bins()];
        Ok(NullableContinuousDataDrift {
            baseline,
            rt_bins,
            n: 0_f64,
            null_n: 0_f64,
        })
    }
}

impl<T: Float + Send + Sync> NullableContinuousDataDrift<T> {
    /// Construct a new instance from a nullable baseline dataset. `None` and `Some(NaN)` values
    /// are filtered out before bin edges are derived. The null fraction is recorded and returned
    /// alongside each drift result.
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if no non-null values remain after filtering,
    /// or [`DriftError::NaNValueError`] if any `Some(NaN)` value is present.
    pub fn new_from_baseline(
        quantile_type: Option<QuantileType>,
        bl_slice: &[Option<T>],
    ) -> Result<NullableContinuousDataDrift<T>, DriftError> {
        let baseline = NullableBaselineContinuousBins::new(bl_slice, quantile_type)?;
        let rt_bins = vec![0_f64; baseline.baseline_bins().len()];
        Ok(NullableContinuousDataDrift {
            baseline,
            rt_bins,
            n: 0_f64,
            null_n: 0_f64,
        })
    }

    /// Compute drift between the baseline and the provided runtime dataset. `None` and `Some(NaN)`
    /// values are treated as null and excluded from the drift computation. The null fraction of
    /// the runtime slice is reported in the returned [`NullableDriftComputation`].
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if `runtime_data` is empty.
    pub fn compute_drift(
        &mut self,
        runtime_data: &[Option<T>],
        drift_type: ContinuousDriftType,
    ) -> Result<NullableDriftComputation<ContinuousDriftType>, DriftError> {
        self.build_rt_hist(runtime_data)?;
        let null_percentage = self.null_n / self.n;
        let drift = compute_drift_continuous(self, drift_type);
        self.clear_rt();
        Ok(NullableDriftComputation {
            drift,
            null_percentage,
        })
    }

    /// Compute drift for multiple metrics in a single call. More efficient than calling
    /// [`compute_drift`] in a loop as runtime binning is performed once.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if `runtime_data` is empty.
    ///
    /// [`compute_drift`]: NullableContinuousDataDrift::compute_drift
    pub fn compute_drift_multiple_criteria(
        &mut self,
        runtime_data: &[Option<T>],
        drift_types: &[ContinuousDriftType],
    ) -> Result<NullableDriftComputationMulti<ContinuousDriftType>, DriftError> {
        self.build_rt_hist(runtime_data)?;
        let null_percentage = self.null_n / self.n;
        let drift = compute_drift_continuous_multi(self, drift_types);
        self.clear_rt();
        Ok(NullableDriftComputationMulti {
            drift,
            null_percentage,
        })
    }

    /// Replace the baseline with a new dataset, preserving the original [`QuantileType`]. All
    /// runtime bins are cleared.
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if no non-null values remain after filtering.
    pub fn reset_baseline(&mut self, baseline_slice: &[Option<T>]) -> Result<(), DriftError> {
        self.baseline.reset(baseline_slice)?;
        self.rt_bins = vec![0_f64; self.baseline.baseline_bins().len()];
        Ok(())
    }

    #[inline]
    fn build_rt_hist(&mut self, runtime_data: &[Option<T>]) -> Result<(), DriftError> {
        if runtime_data.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        self.n = runtime_data.len() as f64;
        let (bins, null_count) = compute_dataset_from_bins_continuous_null_parallel(
            runtime_data,
            &self.baseline.bin_edges(),
        );
        self.null_n = null_count as f64;
        self.rt_bins = bins;
        Ok(())
    }

    fn clear_rt(&mut self) {
        self.rt_bins.fill(0_f64);
        self.n = 0_f64;
        self.null_n = 0_f64;
    }
}

impl<T: Float> NullableContinuousDataDrift<T> {
    /// The number of histogram bins derived from the baseline dataset.
    pub fn n_bins(&self) -> usize {
        self.baseline.n_bins()
    }

    /// Export the baseline bin proportions. Each value represents the proportion of non-null
    /// baseline samples that fell into the corresponding bin.
    pub fn export_baseline(&self) -> Vec<f64> {
        self.baseline.export_baseline()
    }
}

impl<T: Float + serde::Serialize> NullableContinuousDataDrift<T> {
    pub fn export_baseline_state(self) -> export::NullableContinuousDriftBaselineExport<T> {
        self.baseline.into()
    }
}

#[cfg(test)]
mod continuous_test {
    use super::*;

    #[test]
    fn test_continuous_baseline_builds_expected_bins() {
        let baseline = [1.0, 2.0, 3.0, 4.0];
        let psi = ContinuousDataDrift::new_from_baseline(None, &baseline).unwrap();

        let expected_bins = QuantileType::FreedmanDiaconis.compute_num_bins(&baseline);

        assert_eq!(psi.baseline.bin_edges().len(), expected_bins - 2);
        assert_eq!(psi.rt_bins.len(), expected_bins);
    }

    #[test]
    fn test_continuous_psi_zero_when_no_drift() {
        let baseline = [1.0, 2.0, 3.0, 4.0];
        let mut psi = ContinuousDataDrift::new_from_baseline(None, &baseline).unwrap();
        let runtime = [1.0, 2.0, 3.0, 4.0];

        let drift = psi
            .compute_drift(&runtime, ContinuousDriftType::PopulationStabilityIndex)
            .unwrap();
        assert!(drift.drift_magnitude.abs() < 1e-9);
    }

    #[test]
    fn test_continuous_psi_detects_shift() {
        let baseline = [1.0, 2.0, 3.0, 4.0];
        let mut psi = ContinuousDataDrift::new_from_baseline(None, &baseline).unwrap();
        let runtime = [10.0, 11.0, 12.0, 13.0];
        let drift = psi
            .compute_drift(&runtime, ContinuousDriftType::PopulationStabilityIndex)
            .unwrap();
        assert!(drift.drift_magnitude > 0.5);
    }

    #[test]
    fn continuous_batch_empty_baseline_returns_err() {
        assert!(ContinuousDataDrift::new_from_baseline(None, &[1.0]).is_err());
    }

    #[test]
    fn continuous_batch_nan_baseline_returns_err() {
        assert!(ContinuousDataDrift::new_from_baseline(None, &[1.0, f64::NAN, 3.0]).is_err());
    }

    #[test]
    fn continuous_batch_empty_runtime_returns_err() {
        let mut det =
            ContinuousDataDrift::new_from_baseline(None, &[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        assert!(
            det.compute_drift(&[], ContinuousDriftType::PopulationStabilityIndex)
                .is_err()
        );
    }

    #[test]
    fn continuous_batch_compute_drift_is_stateless() {
        let baseline: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let runtime: Vec<f64> = (10..40).map(|i| i as f64).collect();
        let mut det = ContinuousDataDrift::new_from_baseline(None, &baseline).unwrap();

        let d1 = det
            .compute_drift(&runtime, ContinuousDriftType::PopulationStabilityIndex)
            .unwrap();
        let d2 = det
            .compute_drift(&runtime, ContinuousDriftType::PopulationStabilityIndex)
            .unwrap();
        assert!((d1.drift_magnitude - d2.drift_magnitude).abs() < 1e-12);
    }

    // --- all drift types ---

    #[test]
    fn continuous_batch_all_drift_types_finite_nonnegative() {
        let baseline: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let runtime: Vec<f64> = (20..80).map(|i| i as f64).collect();
        let mut det = ContinuousDataDrift::new_from_baseline(None, &baseline).unwrap();

        for drift_type in [
            ContinuousDriftType::PopulationStabilityIndex,
            ContinuousDriftType::KullbackLeibler,
            ContinuousDriftType::JensenShannon,
            ContinuousDriftType::WassersteinDistance,
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
    fn continuous_batch_reset_baseline_changes_n_bins() {
        let mut det =
            ContinuousDataDrift::new_from_baseline(None, &[1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let old_bins = det.n_bins();

        let large_baseline: Vec<f64> = (0..200).map(|i| i as f64).collect();
        det.reset_baseline(&large_baseline).unwrap();

        assert_ne!(det.n_bins(), old_bins);
        assert_eq!(det.rt_bins.len(), det.n_bins());
    }

    // --- export_baseline ---

    #[test]
    fn continuous_batch_export_baseline_sums_to_one() {
        let baseline: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let det = ContinuousDataDrift::new_from_baseline(None, &baseline).unwrap();
        let sum: f64 = det.export_baseline().iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }
}
