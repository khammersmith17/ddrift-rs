use super::{
    ColumnTypeClass,
    candidate::CandidateColumn,
    schema_view::{SchemaValidationResult, validate_schema},
};
use crate::baseline::table::{BaselineColumn, BaselineTable};
use crate::constants::{ALL_CATEGORICAL_DRIFT_MEASUREMENTS, ALL_CONTINUOUS_DRIFT_MEASUREMENTS};
use crate::core::{
    dataset_view::total::NullableComputationView,
    drift_metrics::{
        self, CategoricalDriftMeasurement, ContinuousDriftMeasurement, DriftContainer,
    },
    error::DriftTableError,
};
use crate::drift::{DriftComputationMulti, NullableDriftComputationMulti};
use crate::table::candidate::CandidateTable;
use ahash::{HashMap, HashMapExt};

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
    if let SchemaValidationResult::Invalid(diff) = validate_schema(baseline_table, candidate_table)
    {
        return Err(DriftTableError::SchemaError(diff));
    }

    let mut result: TableDrift = HashMap::with_capacity(baseline_table.len());
    for (column, baseline_state) in baseline_table.iter() {
        // SAFETY: At this point we know schema is valid and there is parity between the baseline
        // and candidate dataset.
        result.insert(
            column.to_string(),
            compute_column_drift(baseline_state, candidate_table.get_column(column).unwrap()),
        );
    }
    Ok(result)
}

fn compute_column_drift(
    baseline_state: &BaselineColumn,
    candidate_state: &CandidateColumn,
) -> TableDriftComputation {
    let candidate_comp_components = candidate_state.computation_view();
    let baseline_comp_components = baseline_state.computation_view();

    let computation_view = NullableComputationView::new_from_parts(
        baseline_comp_components,
        candidate_comp_components,
    );

    let null_percentage =
        computation_view.candidate_null_count / computation_view.runtime_sample_size();

    match candidate_state.type_class() {
        ColumnTypeClass::Continuous => {
            let drift = continuous_dispatch(&computation_view, &ALL_CONTINUOUS_DRIFT_MEASUREMENTS);
            let inner: NullableDriftComputationMulti<ContinuousDriftMeasurement> =
                NullableDriftComputationMulti {
                    drift,
                    null_percentage,
                };
            TableDriftComputation::Continuous(inner)
        }
        ColumnTypeClass::Categorical => {
            let drift =
                categorical_dispatch(&computation_view, &ALL_CATEGORICAL_DRIFT_MEASUREMENTS);
            let inner: NullableDriftComputationMulti<CategoricalDriftMeasurement> =
                NullableDriftComputationMulti {
                    drift,
                    null_percentage,
                };
            TableDriftComputation::Categorical(inner)
        }
    }
}

fn continuous_dispatch(
    computation_view: &NullableComputationView,
    measurements: &[ContinuousDriftMeasurement],
) -> DriftComputationMulti<ContinuousDriftMeasurement> {
    drift_metrics::compute_drift_continuous_multi(computation_view, measurements)
}

fn categorical_dispatch(
    computation_view: &NullableComputationView,
    measurements: &[CategoricalDriftMeasurement],
) -> DriftComputationMulti<CategoricalDriftMeasurement> {
    drift_metrics::compute_drift_categorical_multi(computation_view, measurements)
}
