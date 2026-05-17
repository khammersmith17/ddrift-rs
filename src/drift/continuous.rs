use super::{
    DecayModeMark, DriftComputation, FlushModeMark, NullableDriftComputation,
    NullableDriftComputationMulti, StreamingDataDriftMark, stream_mode::StreamModeInner,
};
use crate::{
    baseline::continuous::{BaselineContinuousBins, NullableBaselineContinuousBins},
    constants,
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
use ahash::{HashMap, HashMapExt};
use num_traits::Float;
use std::{
    marker::PhantomData,
    num::NonZeroU64,
    time::{Duration, Instant},
};

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
        &self.baseline.baseline_hist
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
        let rt_bins = vec![0_f64; baseline.baseline_hist.len()];
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
        self.rt_bins = compute_dataset_from_bins_continuous(runtime_data, &self.baseline.bin_edges);
        Ok(())
    }

    fn init_runtime_containers(&mut self) {
        let len = self.baseline.baseline_hist.len();
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
        &self.baseline.baseline_hist
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
        let rt_bins = vec![0_f64; baseline.baseline_hist.len()];
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
        self.rt_bins = vec![0_f64; self.baseline.baseline_hist.len()];
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

/// Streaming drift detector for continuous (floating-point) features. Maintains a running
/// histogram of observed runtime data that is compared against a fixed baseline distribution.
///
/// The type parameter `M` controls the window management strategy and is set at construction:
///
/// - [`FlushModeMark`]: accumulates data until a sample count or time cadence threshold is
///   reached, then hard-resets the stream window. Exposes [`flush`] and [`last_flush`].
/// - [`DecayModeMark`]: applies exponential decay α = 0.5^(1/half_life) to all bin counts on
///   each [`compute_drift`] call, giving a recency-weighted distribution with no hard cutoff.
///   Does not expose [`flush`] or [`last_flush`].
///
/// # Bin count
///
/// The number of histogram bins is derived from the baseline data using one of three heuristics,
/// selected via [`QuantileType`]:
///
/// - **[`FreedmanDiaconis`]** *(default)*: `width = 2 * IQR * n^(-1/3)`, `k = ceil((max - min) / width)`.
///   Robust to outliers. Preferred for most use cases.
/// - **[`Scott`]**: `width = 3.49 * σ * n^(-1/3)`. Assumes approximately normal data.
/// - **[`Sturges`]**: `k = floor(ln(n)) + 1`. Log-based rule, best for small datasets.
///
/// [`flush`]: StreamingContinuousDataDrift::flush
/// [`last_flush`]: StreamingContinuousDataDrift::last_flush
/// [`compute_drift`]: StreamingContinuousDataDrift::compute_drift
/// [`FreedmanDiaconis`]: QuantileType::FreedmanDiaconis
/// [`Scott`]: QuantileType::Scott
/// [`Sturges`]: QuantileType::Sturges
pub struct StreamingContinuousDataDrift<T: Float, M> {
    baseline: BaselineContinuousBins<T>,
    stream_bins: Vec<f64>,
    total_stream_size: f64,
    mode: StreamModeInner,
    _mark: PhantomData<(T, M)>,
}

impl<T: Float, M> DriftContainer for StreamingContinuousDataDrift<T, M> {
    fn baseline_bins(&self) -> &[f64] {
        &self.baseline.baseline_hist
    }

    fn runtime_bins(&self) -> &[f64] {
        &self.stream_bins
    }

    fn runtime_sample_size(&self) -> f64 {
        self.total_stream_size
    }

    fn baseline_sample_size(&self) -> f64 {
        self.baseline.population_size()
    }
}

impl<T: Float + serde::de::DeserializeOwned> StreamingContinuousDataDrift<T, DecayModeMark> {
    pub fn new_from_base_export(
        export: export::StreamingContinuousBaseExport<T>,
    ) -> Result<StreamingContinuousDataDrift<T, DecayModeMark>, DriftExportError> {
        let export::StreamingContinuousBaseExport {
            baseline: baseline_export,
            stream_mode,
        } = export;
        let mode: StreamModeInner = stream_mode.into();
        let baseline = BaselineContinuousBins::new_from_export(baseline_export)?;

        if matches!(mode, StreamModeInner::Flush { .. }) {
            return Err(DriftExportError::InvalidDriftMode);
        }
        let n_bins = baseline.n_bins();
        Ok(StreamingContinuousDataDrift {
            baseline,
            stream_bins: vec![0_f64; n_bins],
            total_stream_size: 0_f64,
            mode,
            _mark: PhantomData,
        })
    }

    pub fn new_from_stateful_export(
        export: export::StreamingContinuousStatefulExport<T>,
    ) -> Result<StreamingContinuousDataDrift<T, DecayModeMark>, DriftExportError> {
        let export::StreamingContinuousStatefulExport {
            baseline: baseline_export,
            stream_mode,
            stream_bins,
        } = export;
        let mode: StreamModeInner = stream_mode.into();
        let baseline = BaselineContinuousBins::new_from_export(baseline_export)?;

        if matches!(mode, StreamModeInner::Flush { .. }) {
            return Err(DriftExportError::InvalidDriftMode);
        }
        Ok(StreamingContinuousDataDrift {
            baseline,
            stream_bins,
            total_stream_size: 0_f64,
            mode,
            _mark: PhantomData,
        })
    }
}

impl<T: Float + Send + Sync> StreamingContinuousDataDrift<T, DecayModeMark> {
    /// Construct a decay-mode stream. On each [`compute_drift`] or
    /// [`compute_drift_multiple_criteria`] call, all bin counts and `total_stream_size` are
    /// multiplied by α = 0.5^(1/`half_life`), where `half_life` is the number of seconds after
    /// which a sample's weight is halved. Older data is continuously down-weighted rather than
    /// discarded, giving a recency-weighted view of the distribution with no hard resets.
    ///
    /// `quantile_type` controls histogram bin count. See [`ContinuousDataDrift::new_from_baseline`]
    /// for a full description of each option. Pass `None` to use [`FreedmanDiaconis`] (default).
    ///
    /// `half_life_opt`: decay half-life in seconds. A shorter half-life makes the signal more
    /// sensitive to recent shifts at the cost of higher variance — older data loses influence
    /// quickly. A longer half-life produces a smoother, more stable signal that responds slowly
    /// to new patterns. Defaults to 86,400 (24 hours), meaning a sample's contribution is halved
    /// after 24 hours worth of [`compute_drift`] calls.
    ///
    /// When computing multiple drift metrics on the same accumulated state, use
    /// [`compute_drift_multiple_criteria`] — decay is applied once before all metrics are
    /// evaluated. Calling [`compute_drift`] in a loop applies decay on each call.
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if the baseline has fewer than 2 elements.
    ///
    /// [`FreedmanDiaconis`]: QuantileType::FreedmanDiaconis
    /// [`compute_drift`]: StreamingContinuousDataDrift::compute_drift
    /// [`compute_drift_multiple_criteria`]: StreamingContinuousDataDrift::compute_drift_multiple_criteria
    pub fn new_decay(
        baseline_data: &[T],
        quantile_type: Option<QuantileType>,
        half_life_opt: Option<NonZeroU64>,
    ) -> Result<StreamingContinuousDataDrift<T, DecayModeMark>, DriftError> {
        let baseline =
            BaselineContinuousBins::new(baseline_data, quantile_type.unwrap_or_default())?;
        let bl_hist_len = baseline.baseline_hist.len();
        let stream_bins: Vec<f64> = vec![0_f64; bl_hist_len];
        let half_life =
            half_life_opt.unwrap_or(NonZeroU64::new(constants::DEFAULT_DECAY_HALF_LIFE).unwrap());
        let mode = StreamModeInner::ExponentialDecay(0.5_f64.powf(1_f64 / half_life.get() as f64));

        Ok(StreamingContinuousDataDrift {
            stream_bins,
            baseline,
            total_stream_size: 0_f64,
            mode,
            _mark: PhantomData,
        })
    }
}

impl<T: Float> StreamingContinuousDataDrift<T, DecayModeMark> {
    /// Compute drift between the accumulated stream and the baseline. Applies exponential decay
    /// to all bin counts before computing, down-weighting older data by α = 0.5^(1/half_life).
    ///
    /// To compute multiple metrics on the same decayed state, use
    /// [`compute_drift_multiple_criteria`] instead. Each call to this method applies decay once,
    /// so calling it repeatedly for different metrics will compound the decay.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if no data has been accumulated.
    ///
    /// [`compute_drift_multiple_criteria`]: StreamingContinuousDataDrift::compute_drift_multiple_criteria
    pub fn compute_drift(
        &mut self,
        drift_type: ContinuousDriftType,
    ) -> Result<DriftComputation<ContinuousDriftType>, DriftError> {
        if self.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        self.apply_decay();
        Ok(compute_drift_continuous(self, drift_type))
    }

    /// Compute multiple drift metrics against the accumulated stream in a single call. Decay is
    /// applied once before all metrics are evaluated, ensuring all results reflect the same
    /// decayed state. Prefer this over calling [`compute_drift`] in a loop when multiple metrics
    /// are needed simultaneously.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if no data has been accumulated.
    ///
    /// [`compute_drift`]: StreamingContinuousDataDrift::compute_drift
    pub fn compute_drift_multiple_criteria(
        &mut self,
        drift_types: &[ContinuousDriftType],
    ) -> Result<Vec<DriftComputation<ContinuousDriftType>>, DriftError> {
        if self.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        self.apply_decay();
        Ok(compute_drift_continuous_multi(self, drift_types))
    }

    fn apply_decay(&mut self) {
        let StreamModeInner::ExponentialDecay(decay_factor) = self.mode else {
            unreachable!()
        };
        for bin in self.stream_bins.iter_mut() {
            *bin = (*bin * decay_factor).floor();
        }
        self.total_stream_size = (self.total_stream_size * decay_factor).floor();
    }

    /// Push a single example into the stream.
    #[inline]
    pub fn update_stream(&mut self, runtime_example: T) {
        let idx = self.baseline.resolve_bin(runtime_example);

        self.stream_bins[idx] += 1_f64;
        self.total_stream_size += 1_f64;
    }

    /// Push a batch of examples into the stream.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if the slice is empty.
    pub fn update_stream_batch(&mut self, runtime_slice: &[T]) -> Result<(), DriftError> {
        if runtime_slice.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }

        for item in runtime_slice {
            self.update_stream(*item)
        }

        Ok(())
    }
}

impl<T: Float + serde::Serialize> StreamingContinuousDataDrift<T, DecayModeMark> {
    pub fn export_baseline_state(self) -> export::StreamingContinuousBaseExport<T> {
        let baseline: export::ContinuousDriftBaselineExport<T> = self.baseline.into();

        export::StreamingContinuousBaseExport {
            baseline,
            stream_mode: self.mode.into(),
        }
    }

    pub fn export_stream_state(self) -> export::StreamingContinuousStatefulExport<T> {
        let baseline: export::ContinuousDriftBaselineExport<T> = self.baseline.into();

        export::StreamingContinuousStatefulExport {
            baseline,
            stream_bins: self.stream_bins,
            stream_mode: self.mode.into(),
        }
    }
}

impl<T: Float + serde::de::DeserializeOwned> StreamingContinuousDataDrift<T, FlushModeMark> {
    pub fn new_from_base_export(
        export: export::StreamingContinuousBaseExport<T>,
    ) -> Result<StreamingContinuousDataDrift<T, FlushModeMark>, DriftExportError> {
        let export::StreamingContinuousBaseExport {
            baseline: baseline_export,
            stream_mode,
        } = export;
        let mode: StreamModeInner = stream_mode.into();
        let baseline = BaselineContinuousBins::new_from_export(baseline_export)?;

        if matches!(mode, StreamModeInner::ExponentialDecay(_)) {
            return Err(DriftExportError::InvalidDriftMode);
        }
        let n_bins = baseline.n_bins();
        Ok(StreamingContinuousDataDrift {
            baseline,
            stream_bins: vec![0_f64; n_bins],
            total_stream_size: 0_f64,
            mode,
            _mark: PhantomData,
        })
    }

    pub fn new_from_stateful_export(
        export: export::StreamingContinuousStatefulExport<T>,
    ) -> Result<StreamingContinuousDataDrift<T, FlushModeMark>, DriftExportError> {
        let export::StreamingContinuousStatefulExport {
            baseline: baseline_export,
            stream_mode,
            stream_bins,
        } = export;
        let mode: StreamModeInner = stream_mode.into();
        let baseline = BaselineContinuousBins::new_from_export(baseline_export)?;

        if matches!(mode, StreamModeInner::ExponentialDecay(_)) {
            return Err(DriftExportError::InvalidDriftMode);
        }
        Ok(StreamingContinuousDataDrift {
            baseline,
            stream_bins,
            total_stream_size: 0_f64,
            mode,
            _mark: PhantomData,
        })
    }
}

impl<T: Float + Send + Sync> StreamingContinuousDataDrift<T, FlushModeMark> {
    /// Construct a flush-mode stream. The stream accumulates data until either `flush_size_opt`
    /// samples have been observed or `flush_cadence_opt` has elapsed since the last flush —
    /// whichever is reached first — at which point all accumulated runtime data is cleared and
    /// the window restarts fresh.
    ///
    /// `quantile_type` controls histogram bin count. See [`ContinuousDataDrift::new_from_baseline`]
    /// for a full description of each option. Pass `None` to use [`FreedmanDiaconis`] (default).
    ///
    /// `flush_size_opt`: number of accumulated samples that triggers an automatic flush. A lower
    /// value means more frequent resets and a more responsive signal, but each window contains
    /// fewer samples making the drift estimate noisier. Defaults to 1,000,000.
    ///
    /// `flush_cadence_opt`: time elapsed since the last flush that triggers an automatic flush,
    /// regardless of sample count. The time check is amortized over batches of 256 pushes to
    /// avoid reading the clock on every sample. Defaults to 86,400 seconds (24 hours).
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if the baseline has fewer than 2 elements.
    ///
    /// [`FreedmanDiaconis`]: QuantileType::FreedmanDiaconis
    pub fn new_flush(
        baseline_data: &[T],
        quantile_type: Option<QuantileType>,
        flush_size_opt: Option<u64>,
        flush_cadence_opt: Option<Duration>,
    ) -> Result<StreamingContinuousDataDrift<T, FlushModeMark>, DriftError> {
        let baseline =
            BaselineContinuousBins::new(baseline_data, quantile_type.unwrap_or_default())?;
        let bl_hist_len = baseline.baseline_hist.len();
        let stream_bins: Vec<f64> = vec![0_f64; bl_hist_len];
        let flush_size = flush_size_opt.unwrap_or(constants::DEFAULT_MAX_STREAM_SIZE);
        let cadence =
            flush_cadence_opt.unwrap_or(Duration::new(constants::DEFAULT_STREAM_FLUSH_CADENCE, 0));
        let mode = StreamModeInner::Flush {
            size: flush_size as f64,
            cadence,
            last_flush_ts: Instant::now(),
        };

        Ok(StreamingContinuousDataDrift {
            stream_bins,
            baseline,
            total_stream_size: 0_f64,
            mode,
            _mark: PhantomData,
        })
    }
}

impl<T: Float> StreamingContinuousDataDrift<T, FlushModeMark> {
    /// Compute drift between the accumulated stream and the baseline.
    ///
    /// To compute multiple metrics on the same accumulated state, use
    /// [`compute_drift_multiple_criteria`] instead.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if no data has been accumulated since the last
    /// flush.
    ///
    /// [`compute_drift_multiple_criteria`]: StreamingContinuousDataDrift::compute_drift_multiple_criteria
    pub fn compute_drift(
        &mut self,
        drift_type: ContinuousDriftType,
    ) -> Result<DriftComputation<ContinuousDriftType>, DriftError> {
        if self.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        Ok(compute_drift_continuous(self, drift_type))
    }

    /// Compute multiple drift metrics against the accumulated stream in a single call. Prefer
    /// this over calling [`compute_drift`] in a loop when multiple metrics are needed
    /// simultaneously.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if no data has been accumulated since the last
    /// flush.
    ///
    /// [`compute_drift`]: StreamingContinuousDataDrift::compute_drift
    pub fn compute_drift_multiple_criteria(
        &mut self,
        drift_types: &[ContinuousDriftType],
    ) -> Result<Vec<DriftComputation<ContinuousDriftType>>, DriftError> {
        if self.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }

        Ok(compute_drift_continuous_multi(self, drift_types))
    }

    /// Push a single example into the stream. A flush is triggered before the item is recorded
    /// if the flush size or cadence threshold has been reached, starting a fresh window.
    #[inline]
    pub fn update_stream(&mut self, runtime_example: T) {
        let idx = self.baseline.resolve_bin(runtime_example);
        if self.mode.needs_flush(self.total_stream_size) {
            self.flush()
        }
        self.stream_bins[idx] += 1_f64;
        self.total_stream_size += 1_f64;
    }

    /// Push a batch of examples into the stream. Each item is checked against flush thresholds
    /// individually, so a flush may occur mid-batch if the size threshold is crossed.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if the slice is empty.
    pub fn update_stream_batch(&mut self, runtime_slice: &[T]) -> Result<(), DriftError> {
        if runtime_slice.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        runtime_slice
            .iter()
            .for_each(|item| self.update_stream(*item));

        Ok(())
    }

    /// Manually flush the stream, clearing all accumulated runtime data. The baseline is not
    /// affected. The flush timestamp is reset so the cadence timer restarts from this point.
    pub fn flush(&mut self) {
        self.flush_runtime_stream();
        self.mode.perform_flush();
    }

    /// Returns the number of seconds elapsed since the last flush.
    pub fn last_flush(&self) -> u64 {
        self.mode.last_flush()
    }
}

impl<T: Float + serde::Serialize> StreamingContinuousDataDrift<T, FlushModeMark> {
    pub fn export_baseline_state(self) -> export::StreamingContinuousBaseExport<T> {
        let baseline: export::ContinuousDriftBaselineExport<T> = self.baseline.into();

        export::StreamingContinuousBaseExport {
            baseline,
            stream_mode: self.mode.into(),
        }
    }

    pub fn export_stream_state(self) -> export::StreamingContinuousStatefulExport<T> {
        let baseline: export::ContinuousDriftBaselineExport<T> = self.baseline.into();

        export::StreamingContinuousStatefulExport {
            baseline,
            stream_bins: self.stream_bins,
            stream_mode: self.mode.into(),
        }
    }
}

#[allow(private_bounds)]
impl<T: Float, M: StreamingDataDriftMark> StreamingContinuousDataDrift<T, M> {
    /// Returns `true` if no data has been accumulated since construction or the last flush.
    pub fn is_empty(&self) -> bool {
        self.stream_bins.iter().sum::<f64>() == 0_f64
    }

    /// The number of samples accumulated in the stream since the last flush. In decay mode this
    /// reflects the effective (decayed) sample count rather than the raw push count.
    pub fn total_samples(&self) -> usize {
        self.total_stream_size as usize
    }

    /// The number of histogram bins derived from the baseline dataset.
    pub fn n_bins(&self) -> usize {
        self.baseline.n_bins()
    }

    /// Export a point-in-time snapshot of the stream state as a map with three keys:
    ///
    /// - `"binEdges"`: the histogram bin edge values defining the boundaries between bins.
    /// - `"baselineBins"`: proportional bin distribution of the baseline dataset.
    /// - `"streamBins"`: proportional bin distribution of the currently accumulated stream data.
    ///
    /// All bin values are normalized to proportions in `[0, 1]`.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if no data has been accumulated.
    pub fn export_snapshot(&self) -> Result<HashMap<String, Vec<f64>>, DriftError> {
        if self.total_stream_size == 0_f64 {
            return Err(DriftError::EmptyRuntimeData);
        }
        // determine snapshot shape
        let mut table: HashMap<String, Vec<f64>> = HashMap::with_capacity(3);
        let bin_edges_export = self
            .baseline
            .export_bin_edges()
            .iter()
            .map(|e| e.to_f64().unwrap())
            .collect();
        table.insert("binEdges".into(), bin_edges_export);
        table.insert("baselineBins".into(), self.export_baseline());
        let bin_ratio_snapshot = self
            .stream_bins
            .iter()
            .map(|v| *v / self.total_stream_size)
            .collect();
        table.insert("streamBins".into(), bin_ratio_snapshot);
        Ok(table)
    }

    /// Export the baseline bin proportions. Returns an owned `Vec<f64>`, which contains the
    /// proportional bin distribution present in the baseline set, and thus what all drift metrics
    /// are computed with respect to.
    pub fn export_baseline(&self) -> Vec<f64> {
        self.baseline.export_baseline()
    }

    // zero out all bins
    fn flush_runtime_stream(&mut self) {
        self.stream_bins.fill(0_f64);
        self.total_stream_size = 0_f64;
    }
}

#[allow(private_bounds)]
impl<T: Float + Send + Sync, M: StreamingDataDriftMark> StreamingContinuousDataDrift<T, M> {
    /// Replace the baseline with a new dataset. The bin count is recomputed from the new data
    /// using the same [`QuantileType`] as construction. All accumulated stream data is cleared
    /// and the flush timestamp is reset.
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if the slice has fewer than 2 elements.
    pub fn reset_baseline(&mut self, baseline_slice: &[T]) -> Result<(), DriftError> {
        self.baseline.reset(baseline_slice)?;
        self.stream_bins = vec![0_f64; self.baseline.baseline_hist.len()];
        self.total_stream_size = 0_f64;
        self.mode.perform_flush();
        Ok(())
    }
}

#[cfg(test)]
mod continuous_tests {
    use super::*;

    #[test]
    fn test_continuous_baseline_builds_expected_bins() {
        let baseline = [1.0, 2.0, 3.0, 4.0];
        let psi = ContinuousDataDrift::new_from_baseline(None, &baseline).unwrap();

        let expected_bins = QuantileType::FreedmanDiaconis.compute_num_bins(&baseline);

        assert_eq!(psi.baseline.bin_edges.len(), expected_bins - 2);
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
    fn test_streaming_continuous_accumulation() {
        let baseline = [1_f64, 2_f64, 3_f64, 3_f64, 4_f64];
        let mut streaming =
            StreamingContinuousDataDrift::new_flush(&baseline, None, None, None).unwrap();

        streaming
            .update_stream_batch(&[1.0, 2.0, 2.0, 3.0, 4.0])
            .unwrap();

        let d1 = streaming
            .compute_drift(ContinuousDriftType::PopulationStabilityIndex)
            .unwrap();
        streaming
            .update_stream_batch(&[3.0, 4.0, 2.0, 2.0, 1.0, 3.0])
            .unwrap();

        let d2 = streaming
            .compute_drift(ContinuousDriftType::PopulationStabilityIndex)
            .unwrap();

        assert!(d1.drift_magnitude.abs() < 1e-9);
        assert!(d2.drift_magnitude.abs() < 1e-2);
        assert_eq!(streaming.total_samples(), 11);
    }

    #[test]
    fn test_streaming_flush() {
        let baseline = [1.0, 2.0, 3.0, 4.0];
        let mut streaming =
            StreamingContinuousDataDrift::new_flush(&baseline, None, None, None).unwrap();

        streaming.update_stream_batch(&[1.0, 2.0, 3.0]).unwrap();
        streaming.flush();

        assert_eq!(streaming.total_samples(), 0);
    }

    // --- error paths ---

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
    fn continuous_streaming_empty_batch_returns_err() {
        let mut s =
            StreamingContinuousDataDrift::new_flush(&[1.0, 2.0, 3.0, 4.0, 5.0], None, None, None)
                .unwrap();
        assert!(s.update_stream_batch(&[]).is_err());
    }

    #[test]
    fn continuous_streaming_compute_on_empty_returns_err() {
        let mut s =
            StreamingContinuousDataDrift::new_flush(&[1.0, 2.0, 3.0, 4.0, 5.0], None, None, None)
                .unwrap();
        assert!(
            s.compute_drift(ContinuousDriftType::PopulationStabilityIndex)
                .is_err()
        );
    }

    // --- statelessness ---

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

    // --- streaming flush-mode extras ---

    #[test]
    fn continuous_streaming_flush_is_empty_after() {
        let baseline: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let mut s = StreamingContinuousDataDrift::new_flush(&baseline, None, None, None).unwrap();
        s.update_stream_batch(&[1.0, 2.0, 3.0]).unwrap();
        assert!(!s.is_empty());
        s.flush();
        assert!(s.is_empty());
    }

    #[test]
    fn continuous_streaming_reset_baseline_clears_stream() {
        let baseline: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let mut s = StreamingContinuousDataDrift::new_flush(&baseline, None, None, None).unwrap();
        s.update_stream_batch(&[10.0, 20.0, 30.0]).unwrap();

        let new_baseline: Vec<f64> = (0..100).map(|i| i as f64).collect();
        s.reset_baseline(&new_baseline).unwrap();

        assert!(s.is_empty());
        assert_eq!(s.total_samples(), 0);
        assert_eq!(s.stream_bins.len(), s.baseline.baseline_hist.len());
    }

    #[test]
    fn continuous_streaming_export_snapshot_empty_returns_err() {
        let baseline: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let s = StreamingContinuousDataDrift::new_flush(&baseline, None, None, None).unwrap();
        assert!(s.export_snapshot().is_err());
    }

    #[test]
    fn continuous_streaming_export_snapshot_has_expected_keys() {
        let baseline: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let mut s = StreamingContinuousDataDrift::new_flush(&baseline, None, None, None).unwrap();
        s.update_stream_batch(&[10.0, 20.0, 30.0]).unwrap();

        let snap = s.export_snapshot().unwrap();
        assert!(snap.contains_key("binEdges"));
        assert!(snap.contains_key("baselineBins"));
        assert!(snap.contains_key("streamBins"));
    }

    // --- decay mode ---

    #[test]
    fn continuous_decay_empty_compute_returns_err() {
        let baseline: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let mut s = StreamingContinuousDataDrift::new_decay(
            &baseline,
            Some(QuantileType::default()),
            Some(std::num::NonZeroU64::new(1).unwrap()),
        )
        .unwrap();
        assert!(
            s.compute_drift(ContinuousDriftType::PopulationStabilityIndex)
                .is_err()
        );
    }

    #[test]
    fn continuous_decay_reduces_total_samples() {
        // half_life=1 → α=0.5, so after one compute_drift call total_samples halves
        let baseline: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let mut s = StreamingContinuousDataDrift::new_decay(
            &baseline,
            Some(QuantileType::default()),
            Some(std::num::NonZeroU64::new(1).unwrap()),
        )
        .unwrap();

        let data: Vec<f64> = (0..100).map(|i| (i % 50) as f64).collect();
        s.update_stream_batch(&data).unwrap();
        assert_eq!(s.total_samples(), 100);

        s.compute_drift(ContinuousDriftType::PopulationStabilityIndex)
            .unwrap();
        assert!(s.total_samples() < 100);
    }

    #[test]
    fn continuous_decay_multiple_criteria_applies_decay_once() {
        // half_life=1 → α=0.5. After compute_drift_multiple_criteria with N metrics,
        // total_samples should be floor(100*0.5)=50, same as a single compute_drift call.
        // Calling compute_drift three times separately would give floor(floor(floor(100*0.5)*0.5)*0.5)=12.
        let baseline: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let half_life = std::num::NonZeroU64::new(1).unwrap();
        let qt = QuantileType::default();

        let mut s_multi =
            StreamingContinuousDataDrift::new_decay(&baseline, Some(qt), Some(half_life)).unwrap();
        let data: Vec<f64> = (0..100).map(|i| (i % 50) as f64).collect();
        s_multi.update_stream_batch(&data).unwrap();
        s_multi
            .compute_drift_multiple_criteria(&[
                ContinuousDriftType::PopulationStabilityIndex,
                ContinuousDriftType::KullbackLeibler,
                ContinuousDriftType::JensenShannon,
            ])
            .unwrap();
        let samples_multi = s_multi.total_samples();

        let qt2 = QuantileType::default();
        let mut s_single =
            StreamingContinuousDataDrift::new_decay(&baseline, Some(qt2), Some(half_life)).unwrap();
        s_single.update_stream_batch(&data).unwrap();
        s_single
            .compute_drift(ContinuousDriftType::PopulationStabilityIndex)
            .unwrap();
        let samples_single = s_single.total_samples();

        assert_eq!(samples_multi, samples_single);
    }
}
