use crate::baseline::table::BaselineTable;
use crate::core::drift_metrics::{CategoricalDriftMeasurement, ContinuousDriftMeasurement};
use crate::table::candidate::CandidateTable;

/*
* Accept baseline and candidate table. Compute column-wise drift.
* Accept a set of continuous and categorical drift measurements and compute all measurements.
* */
