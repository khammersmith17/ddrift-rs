use crate::core::drift_metrics::DriftMetric;
pub mod categorical;
pub mod continuous;

pub enum DriftCheckResult<T: DriftMetric> {
    Passed,
    Failed(Vec<FailedMetricResult<T>>),
}

pub enum FailedMetricResult<T: DriftMetric> {
    Passed,
    Failed { metric: T, drift_delta: f64 },
}

pub enum NullableDriftCheckResult<T: DriftMetric> {
    Passed,
    Failed(Vec<NullableFailedMetricResult<T>>),
}

pub enum NullableFailedMetricResult<T: DriftMetric> {
    Passed,
    Failed { metric: T, delta: f32 },
    Nullfailure { delta: f32 },
}

pub use continuous::{
    ContinuousDriftContract, ContinuousDriftContractBuilder, NullableContinuousDriftContract,
    NullableContinuousDriftContractBuilder,
};

pub use categorical::{
    CategoricalDriftContract, CategoricalDriftContractBuilder, NullableCategoricalDriftContract,
    NullableCategoricalDriftContractBuilder,
};
