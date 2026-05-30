use super::candidate::ArrowCandidateTable;
use crate::baseline::ddrift_arrow::ArrowBaselineTable;
use ahash::{HashMap, HashMapExt};
use arrow::{datatypes::DataType, record_batch::RecordBatch};
use std::collections::{HashSet, hash_set::Difference};

pub(crate) fn validate_schema(
    baseline_schema: &SchemaView,
    candidate_schema: &SchemaView,
) -> SchemaValidationResult {
    baseline_schema.validate(candidate_schema)
}

pub struct InvalidSchemaReport {
    pub invalid_type_columns: Vec<String>,
    pub missing_baseline_columns: Vec<String>,
    pub extra_candidate_columns: Vec<String>,
}

pub(crate) enum SchemaValidationResult {
    Valid,
    Invalid(InvalidSchemaReport),
}

/// Lightweight schema representation of a `[crate::arrow::candidate::ArrowCandidateTable]` or
/// [crate::baseline::arrow::ArrowBaselineTable]` t
pub(crate) struct SchemaView {
    schema: HashMap<String, DataType>,
}

impl SchemaView {
    pub(crate) fn from_baseline_table(table: &ArrowBaselineTable) -> SchemaView {
        let schema: HashMap<String, DataType> = table
            .table
            .iter()
            .map(|(name, entry)| (name.clone(), entry.arrow_type.clone()))
            .collect();
        SchemaView { schema }
    }

    pub(crate) fn from_candidate_table(table: &ArrowCandidateTable) -> SchemaView {
        let schema: HashMap<String, DataType> = table
            .table
            .iter()
            .map(|(name, entry)| (name.clone(), entry.arrow_type.clone()))
            .collect();
        SchemaView { schema }
    }

    pub(crate) fn from_record_batch(batch: &RecordBatch) -> SchemaView {
        let batch_schema = batch.schema();
        let schema: HashMap<String, DataType> = batch_schema
            .fields()
            .iter()
            .map(|field| (field.name().clone(), field.data_type().clone()))
            .collect();
        SchemaView { schema }
    }

    pub(crate) fn validate(&self, candidate_schema: &SchemaView) -> SchemaValidationResult {
        /*
         * 1. Check types like for like.
         * 2. Check Baseline - Candidate
         * 3. Check Candidate - Baseline
         * */

        let mut missing_baseline_columns: Vec<String> = Vec::new();
        let mut invalid_type_columns: Vec<String> = Vec::new();
        self.schema.iter().for_each(|(col_name, baseline_type)| {
            if let Some(candidate_type) = candidate_schema.schema.get(col_name) {
                if candidate_type != baseline_type {
                    invalid_type_columns.push(col_name.to_string())
                }
            } else {
                missing_baseline_columns.push(col_name.to_string())
            }
        });

        let extra_candidate_columns: Vec<String> = candidate_schema
            .schema
            .iter()
            .filter_map(|(col_name, _)| {
                if self.schema.contains_key(col_name) {
                    None
                } else {
                    Some(col_name.to_string())
                }
            })
            .collect();

        if missing_baseline_columns.is_empty()
            && invalid_type_columns.is_empty()
            && extra_candidate_columns.is_empty()
        {
            SchemaValidationResult::Valid
        } else {
            SchemaValidationResult::Invalid(InvalidSchemaReport {
                missing_baseline_columns,
                invalid_type_columns,
                extra_candidate_columns,
            })
        }
    }
}
