pub mod discrete;
pub mod stream_mode;
pub mod streaming;

/// Mode marker for streaming drift types that operate in flush mode. When parameterized with
/// this marker, the stream accumulates data until either a sample size threshold or a time
/// cadence is reached, at which point all accumulated data is cleared and monitoring begins
/// fresh. This mode exposes [`flush`], [`last_flush`], and automatic flush on push.
///
/// [`flush`]: StreamingContinuousDataDrift::flush
/// [`last_flush`]: StreamingContinuousDataDrift::last_flush
pub struct FlushModeMark;

/// Mode marker for streaming drift types that operate in exponential decay mode. When
/// parameterized with this marker, older data is continuously down-weighted on each call to
/// [`compute_drift`] or [`compute_drift_multiple_criteria`] by a decay factor
/// α = 0.5^(1/half_life), where `half_life` is expressed in seconds. Data is never hard-cleared,
/// giving a recency-weighted view of the distribution with no discontinuities. This mode does
/// not expose [`flush`] or [`last_flush`].
///
/// [`compute_drift`]: StreamingContinuousDataDrift::compute_drift
/// [`compute_drift_multiple_criteria`]: StreamingContinuousDataDrift::compute_drift_multiple_criteria
/// [`flush`]: StreamingContinuousDataDrift::flush
/// [`last_flush`]: StreamingContinuousDataDrift::last_flush
pub struct DecayModeMark;

// Marker trait to allow shared behavior across the 2 modes.
// Requires #[allow(private_bounds)] at call sites.
pub(crate) trait StreamingDataDriftMark {}
impl StreamingDataDriftMark for FlushModeMark {}
impl StreamingDataDriftMark for DecayModeMark {}

#[derive(Debug)]
pub struct DriftComputation<T: crate::core::drift_metrics::DriftMeasurement> {
    pub drift_type: T,
    pub drift_magnitude: f64,
}

#[derive(Debug)]
pub struct DriftComputationMulti<T: crate::core::drift_metrics::DriftMeasurement> {
    pub drift: Vec<DriftComputation<T>>,
}

#[derive(Debug)]
pub struct NullableDriftComputation<T: crate::core::drift_metrics::DriftMeasurement> {
    pub drift: DriftComputation<T>,
    pub null_percentage: f64,
}

#[derive(Debug)]
pub struct NullableDriftComputationMulti<T: crate::core::drift_metrics::DriftMeasurement> {
    pub drift: Vec<DriftComputation<T>>,
    pub null_percentage: f64,
}

pub(crate) struct DriftActorComponents<'a> {
    pub(crate) bins: &'a [f64],
    pub(crate) count: usize,
}

pub(crate) struct NullDriftActorComponents<'a> {
    pub(crate) bins: &'a [f64],
    pub(crate) count: usize,
    pub(crate) null_count: usize,
}

pub trait DriftActor<'a> {
    fn quantile_bins(&'a self) -> &'a [f64];
    fn example_count(&self) -> usize;
    fn components(&'a self) -> DriftActorComponents<'a> {
        DriftActorComponents {
            bins: self.quantile_bins(),
            count: self.example_count(),
        }
    }
}

pub trait NullableDriftActor<'a>: DriftActor<'a> {
    fn null_count(&self) -> usize;
    fn nullable_components(&'a self) -> NullDriftActorComponents<'a> {
        NullDriftActorComponents {
            bins: self.quantile_bins(),
            count: self.example_count(),
            null_count: self.null_count(),
        }
    }
}
