use super::schema_view::{SchemaValidationResult, SchemaView, validate_schema};
use crate::baseline::table::BaselineTable;
use crate::core::{
    drift_metrics::{CategoricalDriftMeasurement, ContinuousDriftMeasurement},
    error::DriftTableError,
};
use crate::drift::NullableDriftComputationMulti;
use crate::table::candidate::CandidateTable;
use ahash::HashMap;

/*
* Accept baseline and candidate table. Compute column-wise drift.
* Accept a set of continuous and categorical drift measurements and compute all measurements.
* */

pub enum TableDriftComputation {
    Continuous(NullableDriftComputationMulti<ContinuousDriftMeasurement>),
    Categorical(NullableDriftComputationMulti<CategoricalDriftMeasurement>),
}

pub type TableDrift = HashMap<String, TableDriftComputation>;

pub fn compute_table_drift(
    baseline_table: &BaselineTable,
    candidate_table: &CandidateTable,
) -> Result<TableDrift, DriftTableError> {
    let bl_schema: SchemaView = baseline_table.into();
    let candidate_schema: SchemaView = candidate_table.into();
    if let SchemaValidationResult::Invalid(diff) = validate_schema(&bl_schema, &candidate_schema) {
        return Err(DriftTableError::SchemaError(diff));
    }
    for (column, baseline_state) in baseline_table.iter() {
        // SAFETY: At this point we know its there.
        let candidate = candidate_table.get_column(column).unwrap();
    }

    todo!()
}
