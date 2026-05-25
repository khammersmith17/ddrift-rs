use crate::{
    baseline::categorical::{BaselineCategoricalBins, NullableBaselineCategoricalBins},
    constants,
    core::{
        drift_metrics::{
            CategoricalDriftMeasurement, DriftContainer, compute_drift_categorical,
            compute_drift_categorical_multi,
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
use std::{
    hash::Hash,
    marker::PhantomData,
    num::NonZeroU64,
    time::{Duration, Instant},
};

pub struct NullableStreamingCategoricalDataDrift<T: Hash + Ord + Clone, M> {
    baseline: NullableBaselineCategoricalBins<T>,
    stream_bins: Vec<f64>,
    total_stream_size: f64,
    null_count: f64,
    mode: StreamModeInner,
    _mark: PhantomData<M>,
}

impl<T: Hash + Ord + Clone, M> DriftContainer for NullableStreamingCategoricalDataDrift<T, M> {
    fn baseline_bins(&self) -> &[f64] {
        &self.baseline.baseline_bins
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

impl<T: Hash + Ord + Clone + serde::de::DeserializeOwned>
    NullableStreamingCategoricalDataDrift<T, FlushModeMark>
{
    pub fn new_from_base_export(
        export: export::NullableStreamingCategoricalBaseExport,
    ) -> Result<NullableStreamingCategoricalDataDrift<T, FlushModeMark>, DriftExportError> {
        let export::NullableStreamingCategoricalBaseExport {
            baseline: baseline_export,
            stream_mode,
        } = export;
        let mode: StreamModeInner = stream_mode.into();
        let baseline = NullableBaselineCategoricalBins::new_from_export(baseline_export)?;

        if matches!(mode, StreamModeInner::ExponentialDecay(_)) {
            return Err(DriftExportError::InvalidDriftMode);
        }
        let n_bins = baseline.n_bins();
        Ok(NullableStreamingCategoricalDataDrift {
            baseline,
            stream_bins: vec![0_f64; n_bins],
            total_stream_size: 0_f64,
            null_count: 0_f64,
            mode,
            _mark: PhantomData,
        })
    }

    pub fn new_from_stateful_export(
        export: export::NullableStreamingCategoricalStatefulExport,
    ) -> Result<NullableStreamingCategoricalDataDrift<T, FlushModeMark>, DriftExportError> {
        let export::NullableStreamingCategoricalStatefulExport {
            baseline: baseline_export,
            stream_mode,
            stream_bins,
            total_samples,
            null_samples,
        } = export;
        let mode: StreamModeInner = stream_mode.into();
        let baseline = NullableBaselineCategoricalBins::new_from_export(baseline_export)?;

        if matches!(mode, StreamModeInner::ExponentialDecay(_)) {
            return Err(DriftExportError::InvalidDriftMode);
        }
        Ok(NullableStreamingCategoricalDataDrift {
            baseline,
            stream_bins,
            total_stream_size: total_samples,
            null_count: null_samples,
            mode,
            _mark: PhantomData,
        })
    }
}

impl<T: Hash + Ord + Clone + serde::de::DeserializeOwned>
    NullableStreamingCategoricalDataDrift<T, DecayModeMark>
{
    pub fn new_from_base_export(
        export: export::NullableStreamingCategoricalBaseExport,
    ) -> Result<NullableStreamingCategoricalDataDrift<T, DecayModeMark>, DriftExportError> {
        let export::NullableStreamingCategoricalBaseExport {
            baseline: baseline_export,
            stream_mode,
        } = export;
        let mode: StreamModeInner = stream_mode.into();
        let baseline = NullableBaselineCategoricalBins::new_from_export(baseline_export)?;

        if matches!(mode, StreamModeInner::Flush { .. }) {
            return Err(DriftExportError::InvalidDriftMode);
        }
        let n_bins = baseline.n_bins();
        Ok(NullableStreamingCategoricalDataDrift {
            baseline,
            stream_bins: vec![0_f64; n_bins],
            total_stream_size: 0_f64,
            null_count: 0_f64,
            mode,
            _mark: PhantomData,
        })
    }

    pub fn new_from_stateful_export(
        export: export::NullableStreamingCategoricalStatefulExport,
    ) -> Result<NullableStreamingCategoricalDataDrift<T, DecayModeMark>, DriftExportError> {
        let export::NullableStreamingCategoricalStatefulExport {
            baseline: baseline_export,
            stream_mode,
            stream_bins,
            total_samples,
            null_samples,
        } = export;
        let mode: StreamModeInner = stream_mode.into();
        let baseline = NullableBaselineCategoricalBins::new_from_export(baseline_export)?;

        if matches!(mode, StreamModeInner::Flush { .. }) {
            return Err(DriftExportError::InvalidDriftMode);
        }
        Ok(NullableStreamingCategoricalDataDrift {
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
impl<T: Hash + Ord + Clone + serde::Serialize, M: StreamingDataDriftMark>
    NullableStreamingCategoricalDataDrift<T, M>
{
    pub fn export_baseline_state(
        self,
    ) -> Result<export::NullableStreamingCategoricalBaseExport, serde_json::Error> {
        let baseline: export::NullableCategoricalDriftBaselineExport = self.baseline.try_into()?;
        Ok(export::NullableStreamingCategoricalBaseExport {
            baseline,
            stream_mode: self.mode.into(),
        })
    }

    pub fn export_stream_state(
        self,
    ) -> Result<export::NullableStreamingCategoricalStatefulExport, serde_json::Error> {
        let baseline: export::NullableCategoricalDriftBaselineExport = self.baseline.try_into()?;
        Ok(export::NullableStreamingCategoricalStatefulExport {
            baseline,
            stream_bins: self.stream_bins,
            stream_mode: self.mode.into(),
            total_samples: self.total_stream_size,
            null_samples: self.null_count,
        })
    }
}

impl<T: Hash + Ord + Clone> NullableStreamingCategoricalDataDrift<T, FlushModeMark> {
    /// Construct a flush-mode stream. The stream accumulates data until either `flush_size_opt`
    /// samples have been observed or `flush_cadence_opt` has elapsed since the last flush —
    /// whichever is reached first — at which point all accumulated runtime data is cleared and
    /// the window restarts fresh.
    ///
    /// `flush_size_opt`: number of accumulated samples that triggers an automatic flush. A lower
    /// value means more frequent resets and a more responsive signal, but each window contains
    /// fewer samples making the drift estimate noisier. Defaults to 1,000,000.
    ///
    /// `flush_cadence_opt`: time elapsed since the last flush that triggers an automatic flush,
    /// regardless of sample count. The time check is amortized over batches of 256 pushes to
    /// avoid reading the clock on every sample. Defaults to 86,400 seconds (24 hours).
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if `baseline_data` is empty.
    pub fn new_flush(
        baseline_data: &[Option<T>],
        flush_size_opt: Option<u64>,
        flush_cadence_opt: Option<Duration>,
    ) -> Result<NullableStreamingCategoricalDataDrift<T, FlushModeMark>, DriftError> {
        let baseline = NullableBaselineCategoricalBins::new(baseline_data)?;
        let bl_hist_len = baseline.baseline_bins.len();
        let stream_bins: Vec<f64> = vec![0_f64; bl_hist_len];
        let size = flush_size_opt.unwrap_or(constants::DEFAULT_MAX_STREAM_SIZE);
        let cadence =
            flush_cadence_opt.unwrap_or(Duration::new(constants::DEFAULT_STREAM_FLUSH_CADENCE, 0));
        let mode = StreamModeInner::Flush {
            size: size as f64,
            cadence,
            last_flush_ts: Instant::now(),
        };

        Ok(NullableStreamingCategoricalDataDrift {
            stream_bins,
            baseline,
            total_stream_size: 0_f64,
            null_count: 0_f64,
            mode,
            _mark: PhantomData,
        })
    }

    /// Manually flush the stream, clearing all accumulated runtime data. The baseline is not
    /// affected. The flush timestamp is reset so the cadence timer restarts from this point.
    pub fn flush(&mut self) {
        self.mode
            .perform_flush(&mut self.stream_bins, &mut self.total_stream_size);
    }

    pub fn last_flush(&self) -> u64 {
        self.mode.last_flush()
    }
}

impl<T: Hash + Ord + Clone> NullableStreamingCategoricalDataDrift<T, DecayModeMark> {
    /// Construct a decay-mode stream. On each [`compute_drift`] or
    /// [`compute_drift_multiple_criteria`] call, all bin counts and `total_stream_size` are
    /// multiplied by α = 0.5^(1/`half_life`), where `half_life` is the number of seconds after
    /// which a sample's weight is halved. Older data is continuously down-weighted rather than
    /// discarded, giving a recency-weighted view of the distribution with no hard resets.
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
    /// Returns [`DriftError::EmptyBaselineData`] if `baseline_data` is empty.
    ///
    /// [`compute_drift`]: StreamingCategoricalDataDrift::compute_drift
    /// [`compute_drift_multiple_criteria`]: StreamingCategoricalDataDrift::compute_drift_multiple_criteria
    pub fn new_decay(
        baseline_data: &[Option<T>],
        half_life_opt: Option<NonZeroU64>,
    ) -> Result<NullableStreamingCategoricalDataDrift<T, DecayModeMark>, DriftError> {
        let baseline = NullableBaselineCategoricalBins::new(baseline_data)?;
        let bl_hist_len = baseline.baseline_bins.len();
        let stream_bins: Vec<f64> = vec![0_f64; bl_hist_len];
        let half_life =
            half_life_opt.unwrap_or(NonZeroU64::new(constants::DEFAULT_DECAY_HALF_LIFE).unwrap());
        let mode = StreamModeInner::ExponentialDecay(0.5_f64.powf(1_f64 / half_life.get() as f64));

        Ok(NullableStreamingCategoricalDataDrift {
            stream_bins,
            baseline,
            total_stream_size: 0_f64,
            null_count: 0_f64,
            mode,
            _mark: PhantomData,
        })
    }
}

#[allow(private_bounds)]
impl<T: Hash + Ord + Clone, M: StreamingDataDriftMark> NullableStreamingCategoricalDataDrift<T, M> {
    pub fn compute_drift(
        &mut self,
        drift_type: CategoricalDriftMeasurement,
    ) -> Result<NullableDriftComputation<CategoricalDriftMeasurement>, DriftError> {
        if self.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        self.mode.apply_nullable_decay(
            &mut self.stream_bins,
            &mut self.total_stream_size,
            &mut self.null_count,
        );
        Ok(NullableDriftComputation {
            drift: compute_drift_categorical(self, drift_type),
            null_percentage: self.null_count / self.total_stream_size,
        })
    }

    pub fn compute_drift_multiple_criteria(
        &mut self,
        drift_types: &[CategoricalDriftMeasurement],
    ) -> Result<NullableDriftComputationMulti<CategoricalDriftMeasurement>, DriftError> {
        if self.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }

        self.mode.apply_nullable_decay(
            &mut self.stream_bins,
            &mut self.total_stream_size,
            &mut self.null_count,
        );

        let DriftComputationMulti { drift } = compute_drift_categorical_multi(self, drift_types);

        Ok(NullableDriftComputationMulti {
            drift,
            null_percentage: self.null_count / self.total_stream_size,
        })
    }

    fn check_flush(&mut self) {
        if self
            .mode
            .needs_flush(self.total_stream_size + constants::FLUSH_CHECK_OFFSET as f64)
        {
            self.mode
                .perform_flush(&mut self.stream_bins, &mut self.total_stream_size)
        }
    }

    #[inline]
    fn inner_update_stream<Q>(&mut self, item_opt: Option<&Q>)
    where
        T: std::borrow::Borrow<Q>,
        Q: Ord + Hash + ?Sized,
    {
        if let Some(item) = item_opt {
            self.stream_bins[self.baseline.resolve_bin(item)] += 1_f64;
        } else {
            self.null_count += 1_f64;
        }

        self.total_stream_size += 1_f64;
    }

    /// Push a single label into the stream.
    pub fn update_stream<Q>(&mut self, item_opt: Option<&Q>)
    where
        T: std::borrow::Borrow<Q>,
        Q: Ord + Hash + ?Sized,
    {
        self.check_flush();
        self.inner_update_stream(item_opt)
    }

    /// Push a batch of labels into the stream.
    ///
    /// Returns [`DriftError::EmptyRuntimeData`] if the slice is empty.
    pub fn update_stream_batch(&mut self, runtime_data: &[Option<T>]) -> Result<(), DriftError> {
        if runtime_data.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        self.check_flush();
        runtime_data
            .iter()
            .for_each(|cat| self.inner_update_stream(cat.as_ref()));

        Ok(())
    }

    /// Returns `true` if no data has been accumulated since construction or the last flush.
    pub fn is_empty(&self) -> bool {
        self.stream_bins.iter().sum::<f64>() == 0_f64
    }

    /// Replace the baseline with a new dataset. The bin count is recomputed from the new data —
    /// the number of bins becomes the new cardinality plus one "other" bin. All accumulated
    /// stream data is cleared and the flush timestamp is reset.
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if `new_baseline` is empty.
    pub fn reset_baseline(&mut self, new_baseline: &[Option<T>]) -> Result<(), DriftError> {
        self.baseline.reset(new_baseline)?;
        self.init_stream_bins();
        self.total_stream_size = 0_f64;
        self.mode.touch_flush_ts();
        Ok(())
    }

    /// The number of samples accumulated in the stream since the last flush. In decay mode this
    /// reflects the effective (decayed) sample count rather than the raw push count.
    pub fn total_samples(&self) -> usize {
        self.total_stream_size.floor() as usize
    }

    fn init_stream_bins(&mut self) {
        self.stream_bins = vec![0_f64; self.baseline.baseline_bins.len()]
    }

    pub fn num_bins(&self) -> usize {
        self.stream_bins.len()
    }
}

pub struct StreamingCategoricalDataDrift<T: Hash + Ord + Clone, M> {
    baseline: BaselineCategoricalBins<T>,
    stream_bins: Vec<f64>,
    total_stream_size: f64,
    mode: StreamModeInner,
    _mark: PhantomData<M>,
}

impl<T: Hash + Ord + Clone, M> DriftContainer for StreamingCategoricalDataDrift<T, M> {
    fn baseline_bins(&self) -> &[f64] {
        &self.baseline.baseline_bins
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

impl<T: Hash + Ord + Clone> StreamingCategoricalDataDrift<T, FlushModeMark> {
    /// Construct a flush-mode stream. The stream accumulates data until either `flush_size_opt`
    /// samples have been observed or `flush_cadence_opt` has elapsed since the last flush —
    /// whichever is reached first — at which point all accumulated runtime data is cleared and
    /// the window restarts fresh.
    ///
    /// `flush_size_opt`: number of accumulated samples that triggers an automatic flush. A lower
    /// value means more frequent resets and a more responsive signal, but each window contains
    /// fewer samples making the drift estimate noisier. Defaults to 1,000,000.
    ///
    /// `flush_cadence_opt`: time elapsed since the last flush that triggers an automatic flush,
    /// regardless of sample count. The time check is amortized over batches of 256 pushes to
    /// avoid reading the clock on every sample. Defaults to 86,400 seconds (24 hours).
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if `baseline_data` is empty.
    pub fn new_flush(
        baseline_data: &[T],
        flush_size_opt: Option<u64>,
        flush_cadence_opt: Option<Duration>,
    ) -> Result<StreamingCategoricalDataDrift<T, FlushModeMark>, DriftError> {
        let baseline = BaselineCategoricalBins::new(baseline_data)?;
        let bl_hist_len = baseline.baseline_bins.len();
        let stream_bins: Vec<f64> = vec![0_f64; bl_hist_len];
        let size = flush_size_opt.unwrap_or(constants::DEFAULT_MAX_STREAM_SIZE);
        let cadence =
            flush_cadence_opt.unwrap_or(Duration::new(constants::DEFAULT_STREAM_FLUSH_CADENCE, 0));
        let mode = StreamModeInner::Flush {
            size: size as f64,
            cadence,
            last_flush_ts: Instant::now(),
        };

        Ok(StreamingCategoricalDataDrift {
            stream_bins,
            baseline,
            total_stream_size: 0_f64,
            mode,
            _mark: PhantomData,
        })
    }

    /// Manually flush the stream, clearing all accumulated runtime data. The baseline is not
    /// affected. The flush timestamp is reset so the cadence timer restarts from this point.
    pub fn flush(&mut self) {
        self.flush_runtime_stream();
        self.mode
            .perform_flush(&mut self.stream_bins, &mut self.total_stream_size);
    }

    pub fn last_flush(&self) -> u64 {
        self.mode.last_flush()
    }
}

impl<T: Hash + Ord + Clone + serde::de::DeserializeOwned>
    StreamingCategoricalDataDrift<T, FlushModeMark>
{
    pub fn new_from_base_export(
        export: export::StreamingCategoricalBaseExport,
    ) -> Result<StreamingCategoricalDataDrift<T, FlushModeMark>, DriftExportError> {
        let export::StreamingCategoricalBaseExport {
            baseline: baseline_export,
            stream_mode,
        } = export;
        let mode: StreamModeInner = stream_mode.into();
        let baseline = BaselineCategoricalBins::new_from_export(baseline_export)?;

        if matches!(mode, StreamModeInner::ExponentialDecay(_)) {
            return Err(DriftExportError::InvalidDriftMode);
        }
        let n_bins = baseline.n_bins();
        Ok(StreamingCategoricalDataDrift {
            baseline,
            stream_bins: vec![0_f64; n_bins],
            total_stream_size: 0_f64,
            mode,
            _mark: PhantomData,
        })
    }

    pub fn new_from_stateful_export(
        export: export::StreamingCategoricalStatefulExport,
    ) -> Result<StreamingCategoricalDataDrift<T, FlushModeMark>, DriftExportError> {
        let export::StreamingCategoricalStatefulExport {
            baseline: baseline_export,
            stream_mode,
            stream_bins,
            total_stream_size,
        } = export;
        let mode: StreamModeInner = stream_mode.into();
        let baseline = BaselineCategoricalBins::new_from_export(baseline_export)?;

        if matches!(mode, StreamModeInner::ExponentialDecay(_)) {
            return Err(DriftExportError::InvalidDriftMode);
        }
        Ok(StreamingCategoricalDataDrift {
            baseline,
            stream_bins,
            total_stream_size,
            mode,
            _mark: PhantomData,
        })
    }
}

impl<T: Hash + Ord + Clone + serde::Serialize> StreamingCategoricalDataDrift<T, FlushModeMark> {
    pub fn export_baseline_state(
        self,
    ) -> Result<export::StreamingCategoricalBaseExport, serde_json::Error> {
        let baseline: export::CategoricalDriftBaselineExport = self.baseline.try_into()?;
        Ok(export::StreamingCategoricalBaseExport {
            baseline,
            stream_mode: self.mode.into(),
        })
    }

    pub fn export_stream_state(
        self,
    ) -> Result<export::StreamingCategoricalStatefulExport, serde_json::Error> {
        let baseline: export::CategoricalDriftBaselineExport = self.baseline.try_into()?;
        Ok(export::StreamingCategoricalStatefulExport {
            baseline,
            stream_bins: self.stream_bins,
            stream_mode: self.mode.into(),
            total_stream_size: self.total_stream_size,
        })
    }
}

impl<T: Hash + Ord + Clone + serde::de::DeserializeOwned>
    StreamingCategoricalDataDrift<T, DecayModeMark>
{
    pub fn new_from_base_export(
        export: export::StreamingCategoricalBaseExport,
    ) -> Result<StreamingCategoricalDataDrift<T, DecayModeMark>, DriftExportError> {
        let export::StreamingCategoricalBaseExport {
            baseline: baseline_export,
            stream_mode,
        } = export;
        let mode: StreamModeInner = stream_mode.into();
        let baseline = BaselineCategoricalBins::new_from_export(baseline_export)?;

        if matches!(mode, StreamModeInner::Flush { .. }) {
            return Err(DriftExportError::InvalidDriftMode);
        }
        let n_bins = baseline.n_bins();
        Ok(StreamingCategoricalDataDrift {
            baseline,
            stream_bins: vec![0_f64; n_bins],
            total_stream_size: 0_f64,
            mode,
            _mark: PhantomData,
        })
    }

    pub fn new_from_stateful_export(
        export: export::StreamingCategoricalStatefulExport,
    ) -> Result<StreamingCategoricalDataDrift<T, DecayModeMark>, DriftExportError> {
        let export::StreamingCategoricalStatefulExport {
            baseline: baseline_export,
            stream_mode,
            stream_bins,
            total_stream_size,
        } = export;
        let mode: StreamModeInner = stream_mode.into();
        let baseline = BaselineCategoricalBins::new_from_export(baseline_export)?;

        if matches!(mode, StreamModeInner::Flush { .. }) {
            return Err(DriftExportError::InvalidDriftMode);
        }
        Ok(StreamingCategoricalDataDrift {
            baseline,
            stream_bins,
            total_stream_size,
            mode,
            _mark: PhantomData,
        })
    }
}

impl<T: Hash + Ord + Clone + serde::Serialize> StreamingCategoricalDataDrift<T, DecayModeMark> {
    pub fn export_baseline_state(
        self,
    ) -> Result<export::StreamingCategoricalBaseExport, serde_json::Error> {
        let baseline: export::CategoricalDriftBaselineExport = self.baseline.try_into()?;
        Ok(export::StreamingCategoricalBaseExport {
            baseline,
            stream_mode: self.mode.into(),
        })
    }

    pub fn export_stream_state(
        self,
    ) -> Result<export::StreamingCategoricalStatefulExport, serde_json::Error> {
        let baseline: export::CategoricalDriftBaselineExport = self.baseline.try_into()?;
        Ok(export::StreamingCategoricalStatefulExport {
            baseline,
            stream_bins: self.stream_bins,
            stream_mode: self.mode.into(),
            total_stream_size: self.total_stream_size,
        })
    }
}

impl<T: Hash + Ord + Clone> StreamingCategoricalDataDrift<T, DecayModeMark> {
    /// Construct a decay-mode stream. On each [`compute_drift`] or
    /// [`compute_drift_multiple_criteria`] call, all bin counts and `total_stream_size` are
    /// multiplied by α = 0.5^(1/`half_life`), where `half_life` is the number of seconds after
    /// which a sample's weight is halved. Older data is continuously down-weighted rather than
    /// discarded, giving a recency-weighted view of the distribution with no hard resets.
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
    /// Returns [`DriftError::EmptyBaselineData`] if `baseline_data` is empty.
    ///
    /// [`compute_drift`]: StreamingCategoricalDataDrift::compute_drift
    /// [`compute_drift_multiple_criteria`]: StreamingCategoricalDataDrift::compute_drift_multiple_criteria
    pub fn new_decay(
        baseline_data: &[T],
        half_life_opt: Option<NonZeroU64>,
    ) -> Result<StreamingCategoricalDataDrift<T, DecayModeMark>, DriftError> {
        let baseline = BaselineCategoricalBins::new(baseline_data)?;
        let bl_hist_len = baseline.baseline_bins.len();
        let stream_bins: Vec<f64> = vec![0_f64; bl_hist_len];
        let half_life =
            half_life_opt.unwrap_or(NonZeroU64::new(constants::DEFAULT_DECAY_HALF_LIFE).unwrap());
        let mode = StreamModeInner::ExponentialDecay(0.5_f64.powf(1_f64 / half_life.get() as f64));

        Ok(StreamingCategoricalDataDrift {
            stream_bins,
            baseline,
            total_stream_size: 0_f64,
            mode,
            _mark: PhantomData,
        })
    }
}

#[allow(private_bounds)]
impl<T: Hash + Ord + Clone, M: StreamingDataDriftMark> StreamingCategoricalDataDrift<T, M> {
    pub fn compute_drift(
        &mut self,
        drift_type: CategoricalDriftMeasurement,
    ) -> Result<DriftComputation<CategoricalDriftMeasurement>, DriftError> {
        if self.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        self.mode
            .apply_decay(&mut self.stream_bins, &mut self.total_stream_size);

        Ok(compute_drift_categorical(self, drift_type))
    }

    pub fn compute_drift_multiple_criteria(
        &mut self,
        drift_types: &[CategoricalDriftMeasurement],
    ) -> Result<DriftComputationMulti<CategoricalDriftMeasurement>, DriftError> {
        if self.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }
        self.mode
            .apply_decay(&mut self.stream_bins, &mut self.total_stream_size);
        Ok(compute_drift_categorical_multi(self, drift_types))
    }

    fn check_flush(&mut self) {
        if self
            .mode
            .needs_flush(self.total_stream_size + constants::FLUSH_CHECK_OFFSET as f64)
        {
            self.mode
                .perform_flush(&mut self.stream_bins, &mut self.total_stream_size)
        }
    }

    #[inline]
    fn inner_update_stream<Q>(&mut self, example: &Q)
    where
        T: std::borrow::Borrow<Q>,
        Q: Hash + Ord + ?Sized,
    {
        let idx = self.baseline.resolve_bin(example);
        self.stream_bins[idx] += 1_f64;
        self.total_stream_size += 1_f64;
    }

    pub fn update_stream<Q>(&mut self, example: &Q)
    where
        T: std::borrow::Borrow<Q>,
        Q: Hash + Ord + ?Sized,
    {
        self.check_flush();
        self.inner_update_stream(example);
    }

    pub fn update_stream_batch(&mut self, runtime_data: &[T]) -> Result<(), DriftError> {
        if runtime_data.is_empty() {
            return Err(DriftError::EmptyRuntimeData);
        }

        self.check_flush();
        runtime_data
            .iter()
            .for_each(|cat| self.inner_update_stream(cat));

        Ok(())
    }
    /// Returns `true` if no data has been accumulated since construction or the last flush.
    pub fn is_empty(&self) -> bool {
        self.stream_bins.iter().sum::<f64>() == 0_f64
    }

    /// Replace the baseline with a new dataset. The bin count is recomputed from the new data —
    /// the number of bins becomes the new cardinality plus one "other" bin. All accumulated
    /// stream data is cleared and the flush timestamp is reset.
    ///
    /// Returns [`DriftError::EmptyBaselineData`] if `new_baseline` is empty.
    pub fn reset_baseline(&mut self, new_baseline: &[T]) -> Result<(), DriftError> {
        self.baseline.reset(new_baseline)?;
        self.init_stream_bins();
        self.total_stream_size = 0_f64;
        self.mode.touch_flush_ts();
        Ok(())
    }

    /// The number of samples accumulated in the stream since the last flush. In decay mode this
    /// reflects the effective (decayed) sample count rather than the raw push count.
    pub fn total_samples(&self) -> usize {
        self.total_stream_size.floor() as usize
    }

    fn init_stream_bins(&mut self) {
        self.stream_bins = vec![0_f64; self.baseline.baseline_bins.len()]
    }

    fn flush_runtime_stream(&mut self) {
        self.stream_bins.fill(0_f64);
        self.total_stream_size = 0_f64;
    }

    pub fn num_bins(&self) -> usize {
        self.stream_bins.len()
    }
}

#[cfg(test)]
mod categorical_tests {
    use super::*;

    #[test]
    fn test_streaming_categorical_accumulation() {
        let baseline = ["a", "b"];
        let mut streaming =
            StreamingCategoricalDataDrift::new_flush(&baseline, None, None).unwrap();

        streaming.update_stream_batch(&["a", "b"]).unwrap();
        let d1 = streaming
            .compute_drift(CategoricalDriftMeasurement::PopulationStabilityIndex)
            .unwrap();
        let mut stream = Vec::new();

        for _ in 0..500 {
            stream.push("a")
        }

        for _ in 0..490 {
            stream.push("b")
        }
        streaming.update_stream_batch(&stream).unwrap();
        let d2 = streaming
            .compute_drift(CategoricalDriftMeasurement::PopulationStabilityIndex)
            .unwrap();

        assert_eq!(streaming.total_samples(), 992);
        assert!(d1.drift_magnitude < 1e-9);
        assert!(d2.drift_magnitude < 1e-2);
    }

    #[test]
    fn categorical_streaming_empty_batch_returns_err() {
        let mut s = StreamingCategoricalDataDrift::new_flush(&["a", "b"], None, None).unwrap();
        let empty: &[&str] = &[];
        assert!(s.update_stream_batch(empty).is_err());
    }

    #[test]
    fn categorical_streaming_compute_on_empty_returns_err() {
        let mut s = StreamingCategoricalDataDrift::new_flush(&["a", "b"], None, None).unwrap();
        assert!(
            s.compute_drift(CategoricalDriftMeasurement::PopulationStabilityIndex)
                .is_err()
        );
    }

    #[test]
    fn categorical_streaming_flush_resets_stream() {
        let mut s = StreamingCategoricalDataDrift::new_flush(&["a", "b"], None, None).unwrap();
        s.update_stream_batch(&["a", "b", "a"]).unwrap();
        assert!(!s.is_empty());
        s.flush();
        assert!(s.is_empty());
        assert_eq!(s.total_samples(), 0);
    }

    #[test]
    fn categorical_streaming_novel_label_accumulates_in_other_bin() {
        let mut s = StreamingCategoricalDataDrift::new_flush(&["a", "b"], None, None).unwrap();
        // push only labels not in the baseline
        s.update_stream_batch(&["x", "y", "x", "z"]).unwrap();

        // drift should be elevated since all traffic is in the other bin
        let drift = s
            .compute_drift(CategoricalDriftMeasurement::PopulationStabilityIndex)
            .unwrap();
        assert!(drift.drift_magnitude > 0.5);
    }

    #[test]
    fn categorical_streaming_reset_baseline_clears_stream() {
        let mut s = StreamingCategoricalDataDrift::new_flush(&["a", "b"], None, None).unwrap();
        s.update_stream_batch(&["a", "b"]).unwrap();

        s.reset_baseline(&["x", "y", "z"]).unwrap();
        assert!(s.is_empty());
        assert_eq!(s.total_samples(), 0);
        assert_eq!(s.stream_bins.len(), 4); // 3 labels + other
    }

    // --- decay mode ---

    #[test]
    fn categorical_decay_empty_compute_returns_err() {
        let mut s = StreamingCategoricalDataDrift::<&str, DecayModeMark>::new_decay(
            &["a", "b"],
            Some(std::num::NonZeroU64::new(1).unwrap()),
        )
        .unwrap();
        assert!(
            s.compute_drift(CategoricalDriftMeasurement::PopulationStabilityIndex)
                .is_err()
        );
    }

    #[test]
    fn categorical_decay_reduces_total_samples() {
        // half_life=1 → α=0.5
        let mut s = StreamingCategoricalDataDrift::<&str, DecayModeMark>::new_decay(
            &["a", "b"],
            Some(std::num::NonZeroU64::new(1).unwrap()),
        )
        .unwrap();

        let data: Vec<&str> = (0..100)
            .map(|i| if i % 2 == 0 { "a" } else { "b" })
            .collect();
        s.update_stream_batch(&data).unwrap();
        assert_eq!(s.total_samples(), 100);

        s.compute_drift(CategoricalDriftMeasurement::PopulationStabilityIndex)
            .unwrap();
        assert!(s.total_samples() < 100);
    }

    #[test]
    fn categorical_decay_multiple_criteria_applies_decay_once() {
        let half_life = std::num::NonZeroU64::new(1).unwrap();
        let data: Vec<&str> = (0..100)
            .map(|i| if i % 2 == 0 { "a" } else { "b" })
            .collect();

        let mut s_multi = StreamingCategoricalDataDrift::<&str, DecayModeMark>::new_decay(
            &["a", "b"],
            Some(half_life),
        )
        .unwrap();
        s_multi.update_stream_batch(&data).unwrap();
        s_multi
            .compute_drift_multiple_criteria(&[
                CategoricalDriftMeasurement::PopulationStabilityIndex,
                CategoricalDriftMeasurement::KullbackLeibler,
                CategoricalDriftMeasurement::JensenShannon,
            ])
            .unwrap();
        let samples_multi = s_multi.total_samples();

        let mut s_single = StreamingCategoricalDataDrift::<&str, DecayModeMark>::new_decay(
            &["a", "b"],
            Some(half_life),
        )
        .unwrap();
        s_single.update_stream_batch(&data).unwrap();
        s_single
            .compute_drift(CategoricalDriftMeasurement::PopulationStabilityIndex)
            .unwrap();
        let samples_single = s_single.total_samples();

        assert_eq!(samples_multi, samples_single);
    }
}
