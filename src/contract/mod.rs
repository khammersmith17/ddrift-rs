use crate::core::drift_metrics::DriftMeasurement;
pub mod categorical;
pub mod continuous;
mod drift_check;

pub enum DriftState {
    Healthy,
    PartiallyHealthy,
    Unhealthy,
}

impl DriftState {
    fn from_evaluations<T: DriftMeasurement>(
        results: &[DriftMeasurementEvaluation<T>],
    ) -> DriftState {
        let mut passed = 0_usize;
        let mut failed = 0_usize;
        let n = results.len();

        results.iter().for_each(|r| match r.evaluation_result {
            DriftThresholdEvaluation::Passed => passed += 1,
            DriftThresholdEvaluation::Failed(_) => failed += 1,
        });

        match (passed == n, failed == n) {
            (true, _) => DriftState::Healthy,
            (_, true) => DriftState::Unhealthy,
            _ => DriftState::PartiallyHealthy,
        }
    }
}

pub struct DriftCheck<T: DriftMeasurement> {
    results: Vec<DriftMeasurementEvaluation<T>>,
    state: DriftState,
}

impl<T: DriftMeasurement> DriftCheck<T> {
    fn new(results: Vec<DriftMeasurementEvaluation<T>>) -> DriftCheck<T> {
        let state = DriftState::from_evaluations(&results);
        DriftCheck { results, state }
    }
}

pub struct DriftMeasurementEvaluation<T: DriftMeasurement> {
    metric: T,
    evaluation_result: DriftThresholdEvaluation,
}

pub enum DriftThresholdEvaluation {
    Passed,
    Failed(f64), // Delta between observed and expected drift measurement.
}

pub enum NullThresholdEvaluation {
    Passed,
    Failed(f64),
}

impl NullThresholdEvaluation {
    fn new(expected: f64, observed: f64) -> NullThresholdEvaluation {
        let delta = expected - observed;
        if delta > 0_f64 {
            NullThresholdEvaluation::Failed(delta)
        } else {
            NullThresholdEvaluation::Passed
        }
    }
}

pub struct NullableDriftCheck<T: DriftMeasurement> {
    results: Vec<DriftMeasurementEvaluation<T>>,
    null_state: NullThresholdEvaluation,
    state: DriftState,
}

impl<T: DriftMeasurement> NullableDriftCheck<T> {
    fn new(
        results: Vec<DriftMeasurementEvaluation<T>>,
        null_rate_threshold: f64,
        null_rate_observed: f64,
    ) -> NullableDriftCheck<T> {
        let state = DriftState::from_evaluations(&results);
        let null_state = NullThresholdEvaluation::new(null_rate_threshold, null_rate_observed);
        NullableDriftCheck {
            results,
            null_state,
            state,
        }
    }
}

pub use continuous::{
    ContinuousDriftContract, ContinuousDriftContractBuilder, NullableContinuousDriftContract,
    NullableContinuousDriftContractBuilder,
};

pub use categorical::{
    CategoricalDriftContract, CategoricalDriftContractBuilder, NullableCategoricalDriftContract,
    NullableCategoricalDriftContractBuilder,
};
