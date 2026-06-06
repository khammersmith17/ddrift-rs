use super::{candidate::CandidateTable, datatypes::DriftDataType};
use crate::baseline::table::BaselineTable;
use ahash::HashMap;
use arrow::record_batch::RecordBatch;

pub(crate) fn validate_schema<'a, L, R>(left: &'a L, right: &'a R) -> SchemaValidationResult
where
    SchemaView<'a>: From<&'a L>,
    SchemaView<'a>: From<&'a R>,
{
    let left_schema = SchemaView::from(left);
    let right_schema = SchemaView::from(right);
    left_schema.validate(&right_schema)
}

#[derive(Debug)]
pub struct InvalidSchemaReport {
    pub invalid_type_columns: Vec<String>,
    pub missing_baseline_columns: Vec<String>,
    pub extra_candidate_columns: Vec<String>,
}

pub(crate) enum SchemaValidationResult {
    Valid,
    Invalid(InvalidSchemaReport),
}

/// Lightweight schema representation of a `[crate::table::candidate::ArrowCandidateTable]` or
/// `[crate::baseline::table::ArrowBaselineTable]`.
pub(crate) struct SchemaView<'a> {
    schema: HashMap<&'a str, DriftDataType>,
}

impl<'a> From<&'a BaselineTable> for SchemaView<'a> {
    fn from(table: &'a BaselineTable) -> SchemaView<'a> {
        let schema: HashMap<&'a str, DriftDataType> = table
            .iter()
            .map(|(name, entry)| (name.as_str(), entry.datatype))
            .collect();
        SchemaView { schema }
    }
}

impl<'a> From<&'a CandidateTable<'a>> for SchemaView<'a> {
    fn from(table: &'a CandidateTable<'a>) -> SchemaView<'a> {
        let schema: HashMap<&'a str, DriftDataType> = table
            .iter()
            .map(|(name, entry)| (name.as_str(), entry.datatype))
            .collect();
        SchemaView { schema }
    }
}

impl<'a> From<&'a RecordBatch> for SchemaView<'a> {
    fn from(batch: &'a RecordBatch) -> SchemaView<'a> {
        let schema: HashMap<&'a str, DriftDataType> = batch
            .schema_ref()
            .fields()
            .iter()
            .map(|field| (field.name().as_str(), field.data_type().into()))
            .collect();
        SchemaView { schema }
    }
}

impl<'a> SchemaView<'a> {
    pub(crate) fn validate(&self, candidate_schema: &SchemaView) -> SchemaValidationResult {
        /*
         * 1. Check types like for like.
         * 2. Check Baseline - Candidate
         * 3. Check Candidate - Baseline
         * */

        let mut missing_baseline_columns: Vec<String> = Vec::new();
        let mut invalid_type_columns: Vec<String> = Vec::new();
        self.schema.iter().for_each(|(col_name, baseline_type)| {
            if let Some(candidate_type) = candidate_schema.schema.get(*col_name) {
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
