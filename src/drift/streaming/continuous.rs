use crate::{
    baseline::continuous::{BaselineContinuousBins, NullableBaselineContinuousBins},
    constants,
    core::{
        distribution::QuantileType,
        drift_metrics::{
            ContinuousDriftMeasurement, DriftContainer, compute_drift_continuous,
            compute_drift_continuous_multi,
        },
        error::{DriftError, DriftExportError},
    },
    drift::{
        DecayModeMark, DriftComputation, DriftComputationMulti, FlushModeMark,
        NullableDriftComputation, NullableDriftComputationMulti, StreamingDataDriftMark,
        stream_mode::StreamModeInner,
    },
    export,
};

use num_traits::Float;
use std::{
    marker::PhantomData,
    num::NonZeroU64,
    time::{Duration, Instant},
};

pub struct NullableStreamingContinuousDataDrift<T: Float, M> {
    baseline: NullableBaselineContinuousBins<T>,
    stream_bins: Vec<f64>,
    total_stream_size: f64,
    null_count: f64,
    mode: StreamModeInner,
    _mark: PhantomData<(T, M)>,
}

impl<T: Float, M> DriftContainer for NullableStreamingContinuousDataDrift<T, M> {
    fn baseline_bins(&self) -> &[f64] {
        &self.baseline.baseline_bins()
    }

    fn runtime_bins(&self) -> &[f64] {
        &self.stream_bins
    }

    fn runtime_sample_size(&self) -> f64 {
        self.total_stream_size - self.null_count
    }

    fn baseline_sample_size(&self) -> f64 {
        self.baseline.population_size()
    }
}

impl<T: Float + Send + Sync> NullableStreamingContinuousDataDrift<T, DecayModeMark> {
    pub fn new_decay(
        baseline_data: &[Option<T>],
        quantile_type: Option<QuantileType>,
        half_life_opt: Option<NonZeroU64>,
    ) -> Result<NullableStreamingContinuousDataDrift<T, DecayModeMark>, DriftError> {
        let baseline = NullableBaselineContinuousBins::new(baseline_data, quantile_type)?;
        let bl_hist_len = baseline.baseline_bins().len();
        let stream_bins: Vec<f64> = vec![0_f64; bl_hist_len];
        let half_life =
            half_life_opt.unwrap_or(NonZeroU64::new(constants::DEFAULT_DECAY_HALF_LIFE).unwrap());
        let mode = StreamModeInner::ExponentialDecay(0.5_f64.powf(1_f64 / half_life.get() as f64));

        Ok(NullableStreamingContinuousDataDrift {
            stream_bins,
            baseline,
            total_stream_size: 0_f64,
            null_count: 0_f64,
            mode,
            _mark: PhantomData,
        })
    }
}

impl<T: Float + Send + Sync> NullableStreamingContinuousDataDrift<T, FlushModeMark> {
    pub fn new_flush(
        baseline_data: &[Option<T>],
        quantile_type: Option<QuantileType>,
        flush_size_opt: Option<u64>,
        flush_cadence_opt: Option<Duration>,
    ) -> Result<NullableStreamingContinuousDataDrift<T, FlushModeMark>, DriftError> {
        let baseline = NullableBaselineContinuousBins::new(baseline_data, quantile_type)?;
        let bl_hist_len = baseline.baseline_bins().len();
        let stream_bins: Vec<f64> = vec![0_f64; bl_hist_len];
        let flush_size = flush_size_opt.unwrap_or(constants::DEFAULT_MAX_STREAM_SIZE);
        let cadence =
            flush_cadence_opt.unwrap_or(Duration::new(constants::DEFAULT_STREAM_FLUSH_CADENCE, 0));
        let mode = StreamModeInner::Flush {
            size: flush_size as f64,
            cadence,
            last_flush_ts: Instant::now(),
        };

        Ok(NullableStreamingContinuousDataDrift {
            stream_bins,
            baseline,
            total_stream_size: 0_f64,
            null_count: 0_f64,
            mode,
            _mark: PhantomData,
        })
    }

    pub fn flush(&mut self) {
        self.mode
            .perform_flush(&mut self.stream_bins, &mut self.total_stream_size);
    }

    /// Returns the number of seconds elapsed since the last flush.
    pub fn last_flush(&self) -> u64 {
        self.mode.last_flush()
    }
}

impl<T: Float + serde::de::DeserializeOwned>
    NullableStreamingContinuousDataDrift<T, DecayModeMark>
{
    pub fn new_from_base_export(
        export: export::NullableStreamingContinuousBaseExport<T>,
    ) -> Result<NullableStreamingContinuousDataDrift<T, DecayModeMark>, DriftExportError> {
        let export::NullableStreamingContinuousBaseExport {
            baseline: baseline_export,
            stream_mode,
        } = export;
        let mode: StreamModeInner = stream_mode.into();
        let baseline = NullableBaselineContinuousBins::new_from_export(baseline_export)?;

        if matches!(mode, StreamModeInner::Flush { .. }) {
            return Err(DriftExportError::InvalidDriftMode);
        }
        let n_bins = baseline.n_bins();
        Ok(NullableStreamingContinuousDataDrift {
            baseline,
            stream_bins: vec![0_f64; n_bins],
            total_stream_size: 0_f64,
            null_count: 0_f64,
            mode,
            _mark: PhantomData,
        })
    }

    pub fn new_from_stateful_export(
        export: export::NullableStreamingContinuousStatefulExport<T>,
    ) -> Result<NullableStreamingContinuousDataDrift<T, DecayModeMark>, DriftExportError> {
        let export::NullableStreamingContinuousStatefulExport {
            baseline: baseline_export,
            stream_mode,
            stream_bins,
            total_samples,
            null_samples,
        } = export;
        let mode: StreamModeInner = stream_mode.into();
        let baseline = NullableBaselineContinuousBins::new_from_export(baseline_export)?;

        if matches!(mode, StreamModeInner::Flush { .. }) {
            return Err(DriftExportError::InvalidDriftMode);
        }
        Ok(NullableStreamingContinuousDataDrift {
            baseline,
            stream_bins,
            total_stream_size: total_samples,
            null_count: null_samples,
            mode,
            _mark: PhantomData,
        })
    }
}

impl<T: Float + serde::de::DeserializeOwned>
    NullableStreamingContinuousDataDrift<T, FlushModeMark>
{
    pub fn new_from_base_export(
        export: export::NullableStreamingContinuousBaseExport<T>,
    ) -> Result<NullableStreamingContinuousDataDrift<T, FlushModeMark>, DriftExportError> {
        let export::NullableStreamingContinuousBaseExport {
            baseline: baseline_export,
            stream_mode,
        } = export;
        let mode: StreamModeInner = stream_mode.into();
        let baseline = NullableBaselineContinuousBins::new_from_export(baseline_export)?;

        if matches!(mode, StreamModeInner::ExponentialDecay(_)) {
            return Err(DriftExportError::InvalidDriftMode);
        }
        let n_bins = baseline.n_bins();
        Ok(NullableStreamingContinuousDataDrift {
            baseline,
            stream_bins: vec![0_f64; n_bins],
            total_stream_size: 0_f64,
            null_count: 0_f64,
            mode,
            _mark: PhantomData,
        })
    }

    pub fn new_from_stateful_export(
        export: export::NullableStreamingContinuousStatefulExport<T>,
    ) -> Result<NullableStreamingContinuousDataDrift<T, FlushModeMark>, DriftExportError> {
        let export::NullableStreamingContinuousStatefulExport {
            baseline: baseline_export,
            stream_mode,
            stream_bins,
            total_samples,
            null_samples,
        } = export;
        let mode: StreamModeInner = stream_mode.into();
        let baseline = NullableBaselineContinuousBins::new_from_export(baseline_export)?;

        if matches!(mode, StreamModeInner::ExponentialDecay(_)) {
            return Err(DriftExportError::InvalidDriftMode);
        }
        Ok(NullableStreamingContinuousDataDrift {
            baseline,
            stream_bins,
            total_stream_size: total_samples,
            null_count: null_samples,
            mode,
            _mark: PhantomData,
        })
    }
}

#[allow(private_bounds)]
impl<T: Float + serde::Serialize, M: StreamingDataDriftMark>
    NullableStreamingContinuousDataDrift<T, M>
{
    pub fn export_baseline_state(self) -> export::NullableStreamingContinuousBaseExport<T> {
        let baseline: export::NullableContinuousDriftBaselineExport<T> = self.baseline.into();

        export::NullableStreamingContinuousBaseExport {
            baseline,
            stream_mode: self.mode.into(),
        }
    }

    pub fn export_stream_state(self) -> export::NullableStreamingContinuousStatefulExport<T> {
        let baseline: export::NullableContinuousDriftBaselineExport<T> = self.baseline.into();

        export::NullableStreamingContinuousStatefulExport {
            baseline,
            stream_bins: self.stream_bins,
            stream_mode: self.mode.into(),
            null_samples: self.null_count,
            total_samples: self.total_stream_size,
        }
    }
}

#[allow(private_bounds)]
impl<T: Float, M: StreamingDataDriftMark> NullableStreamingContinuousDataDrift<T, M> {
    pub fn compute_drift(
        &mut self,
        drift_type: ContinuousDriftMeasurement,
    ) -> Result<NullableDriftComputation<ContinuousDriftMeasurement>, DriftError> {
        if self.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        self.mode
            .apply_decay(&mut self.stream_bins, &mut self.total_stream_size);
        let drift = compute_drift_continuous(self, drift_type);
        Ok(NullableDriftComputation {
            drift,
            null_percentage: self.null_count / self.total_stream_size,
        })
    }

    pub fn compute_drift_multiple_criteria(
        &mut self,
        drift_types: &[ContinuousDriftMeasurement],
    ) -> Result<NullableDriftComputationMulti<ContinuousDriftMeasurement>, DriftError> {
        if self.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }

        self.mode
            .apply_decay(&mut self.stream_bins, &mut self.total_stream_size);

        let DriftComputationMulti { drift } = compute_drift_continuous_multi(self, drift_types);
        Ok(NullableDriftComputationMulti {
            drift,
            null_percentage: self.null_count / self.total_stream_size,
        })
    }

    fn check_flush(&mut self) {
        if self.mode.needs_flush(self.total_stream_size) {
            self.mode
                .perform_flush(&mut self.stream_bins, &mut self.total_stream_size)
        }
    }

    #[inline]
    fn inner_update_stream(&mut self, runtime_example: Option<T>) {
        if let Some(idx) = self.baseline.resolve_bin(runtime_example) {
            self.stream_bins[idx] += 1_f64;
            self.total_stream_size += 1_f64;
        } else {
            self.null_count += 1_f64
        }
    }

    pub fn update_stream(&mut self, runtime_example: Option<T>) {
        self.check_flush();
        self.inner_update_stream(runtime_example)
    }

    /// Push a batch of examples into the stream. Each item is checked against flush thresholds
    /// individually, so a flush may occur mid-batch if the size threshold is crossed.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if the slice is empty.
    pub fn update_stream_batch(&mut self, runtime_slice: &[Option<T>]) -> Result<(), DriftError> {
        if runtime_slice.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        self.check_flush();
        runtime_slice
            .iter()
            .for_each(|item| self.inner_update_stream(*item));

        Ok(())
    }

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
        &self.baseline.baseline_bins()
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

#[allow(private_bounds)]
impl<T: Float + serde::Serialize, M: StreamingDataDriftMark> StreamingContinuousDataDrift<T, M> {
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
        let baseline = BaselineContinuousBins::new(baseline_data, quantile_type)?;
        let bl_hist_len = baseline.baseline_bins().len();
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
        let baseline = BaselineContinuousBins::new(baseline_data, quantile_type)?;
        let bl_hist_len = baseline.baseline_bins().len();
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

    pub fn flush(&mut self) {
        self.mode
            .perform_flush(&mut self.stream_bins, &mut self.total_stream_size);
    }

    /// Returns the number of seconds elapsed since the last flush.
    pub fn last_flush(&self) -> u64 {
        self.mode.last_flush()
    }
}

#[allow(private_bounds)]
impl<T: Float, M: StreamingDataDriftMark> StreamingContinuousDataDrift<T, M> {
    pub fn compute_drift(
        &mut self,
        drift_type: ContinuousDriftMeasurement,
    ) -> Result<DriftComputation<ContinuousDriftMeasurement>, DriftError> {
        if self.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        self.mode
            .apply_decay(&mut self.stream_bins, &mut self.total_stream_size);
        Ok(compute_drift_continuous(self, drift_type))
    }

    pub fn compute_drift_multiple_criteria(
        &mut self,
        drift_types: &[ContinuousDriftMeasurement],
    ) -> Result<DriftComputationMulti<ContinuousDriftMeasurement>, DriftError> {
        if self.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }

        self.mode
            .apply_decay(&mut self.stream_bins, &mut self.total_stream_size);

        Ok(compute_drift_continuous_multi(self, drift_types))
    }

    #[inline]
    fn inner_update_stream(&mut self, runtime_example: T) {
        let idx = self.baseline.resolve_bin(runtime_example);
        self.stream_bins[idx] += 1_f64;
        self.total_stream_size += 1_f64;
    }

    pub fn update_stream(&mut self, runtime_example: T) {
        if self.mode.needs_flush(self.total_stream_size) {
            self.mode
                .perform_flush(&mut self.stream_bins, &mut self.total_stream_size)
        }
        self.inner_update_stream(runtime_example)
    }

    /// Push a batch of examples into the stream. Each item is checked against flush thresholds
    /// individually, so a flush may occur mid-batch if the size threshold is crossed.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if the slice is empty.
    pub fn update_stream_batch(&mut self, runtime_slice: &[T]) -> Result<(), DriftError> {
        if runtime_slice.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        if self
            .mode
            .needs_flush(self.total_stream_size + constants::FLUSH_CHECK_OFFSET as f64)
        {
            self.mode
                .perform_flush(&mut self.stream_bins, &mut self.total_stream_size)
        }
        runtime_slice
            .iter()
            .for_each(|item| self.inner_update_stream(*item));

        Ok(())
    }

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
        self.stream_bins = vec![0_f64; self.baseline.baseline_bins().len()];
        self.total_stream_size = 0_f64;
        self.mode.touch_flush_ts();
        Ok(())
    }
}

#[cfg(test)]
mod continuous_tests {
    use super::*;

    #[test]
    fn test_streaming_continuous_accumulation() {
        let baseline = [1_f64, 2_f64, 3_f64, 3_f64, 4_f64];
        let mut streaming =
            StreamingContinuousDataDrift::new_flush(&baseline, None, None, None).unwrap();

        streaming
            .update_stream_batch(&[1.0, 2.0, 2.0, 3.0, 4.0])
            .unwrap();

        let d1 = streaming
            .compute_drift(ContinuousDriftMeasurement::PopulationStabilityIndex)
            .unwrap();
        streaming
            .update_stream_batch(&[3.0, 4.0, 2.0, 2.0, 1.0, 3.0])
            .unwrap();

        let d2 = streaming
            .compute_drift(ContinuousDriftMeasurement::PopulationStabilityIndex)
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
            s.compute_drift(ContinuousDriftMeasurement::PopulationStabilityIndex)
                .is_err()
        );
    }

    // --- statelessness ---

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
        assert_eq!(s.stream_bins.len(), s.baseline.baseline_bins().len());
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
            s.compute_drift(ContinuousDriftMeasurement::PopulationStabilityIndex)
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

        s.compute_drift(ContinuousDriftMeasurement::PopulationStabilityIndex)
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
                ContinuousDriftMeasurement::PopulationStabilityIndex,
                ContinuousDriftMeasurement::KullbackLeibler,
                ContinuousDriftMeasurement::JensenShannon,
            ])
            .unwrap();
        let samples_multi = s_multi.total_samples();

        let qt2 = QuantileType::default();
        let mut s_single =
            StreamingContinuousDataDrift::new_decay(&baseline, Some(qt2), Some(half_life)).unwrap();
        s_single.update_stream_batch(&data).unwrap();
        s_single
            .compute_drift(ContinuousDriftMeasurement::PopulationStabilityIndex)
            .unwrap();
        let samples_single = s_single.total_samples();

        assert_eq!(samples_multi, samples_single);
    }
}
