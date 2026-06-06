pub mod candidate;
pub(crate) mod schema_view;
pub use schema_view::InvalidSchemaReport;
pub mod candidate_drift;
pub(crate) mod datatypes;
pub(crate) mod slice_impl;
pub use datatypes::DriftDataType;

pub enum ColumnTypeClass {
    Continuous,
    Categorical,
}
