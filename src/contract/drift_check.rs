pub mod continuous_check {
    use crate::contract::{
        DriftCheck, NullableDriftCheck,
        continuous::{ContinuousDriftContract, NullableContinuousDriftContract},
    };
    use crate::core::drift_metrics::ContinuousDriftMeasurement;
    use crate::drift::{DriftComputationMulti, NullableDriftComputationMulti};

    pub fn perform_drift_measurement_check(
        contract: &ContinuousDriftContract,
        measurements: &DriftComputationMulti<ContinuousDriftMeasurement>,
    ) -> DriftCheck<ContinuousDriftMeasurement> {
        let &DriftComputationMulti { ref drift } = measurements;
        contract.check(drift)
    }

    pub fn perform_check_nullable(
        contract: &NullableContinuousDriftContract,
        measurements: &NullableDriftComputationMulti<ContinuousDriftMeasurement>,
    ) -> NullableDriftCheck<ContinuousDriftMeasurement> {
        let &NullableDriftComputationMulti {
            ref drift,
            ref null_percentage,
        } = measurements;
        contract.check(*null_percentage, drift.as_ref())
    }
}
