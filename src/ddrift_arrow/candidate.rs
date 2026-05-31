use super::schema_view::{SchemaValidationResult, SchemaView, validate_schema};
use crate::{
    baseline::{
        categorical::NullableBaselineCategoricalBins,
        continuous::NullableBaselineContinuousBins,
        ddrift_arrow::{ArrowBaselineColumn, ArrowBaselineContainer, ArrowBaselineTable},
    },
    core::{
        dataset_view::candidate::{
            NullableCategoricalCandidateView, NullableContinuousCandidateView,
        },
        error::DriftArrowError,
    },
};
use ahash::{HashMap, HashMapExt};
use arrow::{array::Array, datatypes::DataType, record_batch::RecordBatch};
use std::sync::Arc;

fn candidate_arrow_array_string_dispatch<'a>(
    array: Arc<dyn Array>,
    bin_edges: &'a crate::core::bin_edges::CategoricalBinEdges<String>,
) -> Result<NullableCategoricalCandidateView<'a, String>, crate::core::error::DriftError> {
    use super::slice_impl::{StringSlice32, StringSlice64};

    // Match on inner string type to determine concrete slice type.
    // Dispatch follows the same route given either slice dispatch.
    match array.data_type() {
        DataType::Utf8 | DataType::Dictionary(_, _) => {
            let typed = array
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap();
            let slice = StringSlice32::from_array(typed);
            NullableCategoricalCandidateView::from_string_slice(&slice, bin_edges)
        }
        DataType::LargeUtf8 => {
            let typed = array
                .as_any()
                .downcast_ref::<arrow::array::LargeStringArray>()
                .unwrap();
            let slice = StringSlice64::from_array(typed);
            NullableCategoricalCandidateView::from_string_slice(&slice, bin_edges)
        }
        _ => unreachable!(),
    }
}

/* TODO:
* implement a candidate dataset view, that can take the bin edges from the baseline by reference.
* This binds the candidate dataset to the lifetime of the baseline, OK here.
*
* If a user wants a concrete representation of the runtime dataset, they can get that too.
* */

pub enum ArrowCandidateContainer<'a> {
    FloatingPoint32(NullableContinuousCandidateView<'a, f32>),
    FloatingPoint64(NullableContinuousCandidateView<'a, f64>),
    Integer64(NullableCategoricalCandidateView<'a, i64>),
    Integer32(NullableCategoricalCandidateView<'a, i32>),
    Integer16(NullableCategoricalCandidateView<'a, i16>),
    Integer8(NullableCategoricalCandidateView<'a, i8>),
    UnsignedInteger64(NullableCategoricalCandidateView<'a, u64>),
    UnsignedInteger32(NullableCategoricalCandidateView<'a, u32>),
    UnsignedInteger16(NullableCategoricalCandidateView<'a, u16>),
    UnsignedInteger8(NullableCategoricalCandidateView<'a, u8>),
    String(NullableCategoricalCandidateView<'a, String>),
    Boolean(NullableCategoricalCandidateView<'a, bool>),
}

pub struct ArrowCandidateColumn<'a> {
    pub arrow_type: DataType,
    pub container: ArrowCandidateContainer<'a>,
}

impl<'a> ArrowCandidateColumn<'a> {
    // Caller must guarantee schema parity between baseline and array before calling this.
    pub(super) fn from_baseline_and_array(
        baseline: &'a ArrowBaselineColumn,
        array: Arc<dyn Array>,
    ) -> Result<ArrowCandidateColumn<'a>, DriftArrowError> {
        let &ArrowBaselineColumn {
            arrow_type: ref baseline_arrow_type,
            container: ref baseline_container,
        } = baseline;

        let candidate_arrow_type = array.data_type();
        debug_assert_eq!(baseline_arrow_type, candidate_arrow_type);

        let container = match baseline_container {
            ArrowBaselineContainer::FloatingPoint32(bl_bins) => {
                let &NullableBaselineContinuousBins { ref bin_edges, .. } = bl_bins;
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow::array::Float32Array>()
                    .unwrap();
                let inner = NullableContinuousCandidateView::arrow_from_bin_edges(
                    typed_array.values(),
                    bin_edges,
                    typed_array.nulls(),
                )?;
                ArrowCandidateContainer::FloatingPoint32(inner)
            }
            ArrowBaselineContainer::FloatingPoint64(bl_bins) => {
                let &NullableBaselineContinuousBins { ref bin_edges, .. } = bl_bins;
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .unwrap();
                let inner = NullableContinuousCandidateView::arrow_from_bin_edges(
                    typed_array.values(),
                    bin_edges,
                    typed_array.nulls(),
                )?;
                ArrowCandidateContainer::FloatingPoint64(inner)
            }
            ArrowBaselineContainer::Integer8(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow::array::Int8Array>()
                    .unwrap();
                let inner = NullableCategoricalCandidateView::arrow_from_bin_edges(
                    typed_array.values(),
                    bin_edges,
                    typed_array.nulls(),
                )?;
                ArrowCandidateContainer::Integer8(inner)
            }
            ArrowBaselineContainer::Integer16(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow::array::Int16Array>()
                    .unwrap();
                let inner = NullableCategoricalCandidateView::arrow_from_bin_edges(
                    typed_array.values(),
                    bin_edges,
                    typed_array.nulls(),
                )?;
                ArrowCandidateContainer::Integer16(inner)
            }
            ArrowBaselineContainer::Integer32(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow::array::Int32Array>()
                    .unwrap();
                let inner = NullableCategoricalCandidateView::arrow_from_bin_edges(
                    typed_array.values(),
                    bin_edges,
                    typed_array.nulls(),
                )?;
                ArrowCandidateContainer::Integer32(inner)
            }
            ArrowBaselineContainer::Integer64(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow::array::Int64Array>()
                    .unwrap();
                let inner = NullableCategoricalCandidateView::arrow_from_bin_edges(
                    typed_array.values(),
                    bin_edges,
                    typed_array.nulls(),
                )?;
                ArrowCandidateContainer::Integer64(inner)
            }
            ArrowBaselineContainer::UnsignedInteger8(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow::array::UInt8Array>()
                    .unwrap();
                let inner = NullableCategoricalCandidateView::arrow_from_bin_edges(
                    typed_array.values(),
                    bin_edges,
                    typed_array.nulls(),
                )?;
                ArrowCandidateContainer::UnsignedInteger8(inner)
            }
            ArrowBaselineContainer::UnsignedInteger16(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow::array::UInt16Array>()
                    .unwrap();
                let inner = NullableCategoricalCandidateView::arrow_from_bin_edges(
                    typed_array.values(),
                    bin_edges,
                    typed_array.nulls(),
                )?;
                ArrowCandidateContainer::UnsignedInteger16(inner)
            }
            ArrowBaselineContainer::UnsignedInteger32(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow::array::UInt32Array>()
                    .unwrap();
                let inner = NullableCategoricalCandidateView::arrow_from_bin_edges(
                    typed_array.values(),
                    bin_edges,
                    typed_array.nulls(),
                )?;
                ArrowCandidateContainer::UnsignedInteger32(inner)
            }
            ArrowBaselineContainer::UnsignedInteger64(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow::array::UInt64Array>()
                    .unwrap();
                let inner = NullableCategoricalCandidateView::arrow_from_bin_edges(
                    typed_array.values(),
                    bin_edges,
                    typed_array.nulls(),
                )?;
                ArrowCandidateContainer::UnsignedInteger64(inner)
            }
            ArrowBaselineContainer::String(bl_bins) => {
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let inner = candidate_arrow_array_string_dispatch(array.clone(), bin_edges)?;
                ArrowCandidateContainer::String(inner)
            }
            ArrowBaselineContainer::Boolean(bl_bins) => {
                use super::slice_impl::BooleanSlice;
                let &NullableBaselineCategoricalBins { ref bin_edges, .. } = bl_bins;
                let typed_array = array
                    .as_any()
                    .downcast_ref::<arrow::array::BooleanArray>()
                    .unwrap();
                let slice = BooleanSlice::from_array(&typed_array);
                let inner = NullableCategoricalCandidateView::from_bool_slice(&slice, bin_edges)?;
                ArrowCandidateContainer::Boolean(inner)
            }
        };
        Ok(ArrowCandidateColumn {
            arrow_type: candidate_arrow_type.clone(),
            container,
        })
    }
}

pub struct ArrowCandidateTable<'a> {
    pub table: HashMap<String, ArrowCandidateColumn<'a>>,
}

/*
* All entries start with a validation.
* This provides three benefits:
*   First, we know the schema is correct and has parity.
*   Second, we know there are not unsupported types in the candidate table.
*   Third, we know downstream there are no structural error cases.
* */

impl<'a> ArrowCandidateTable<'a> {
    pub fn from_record_batch(
        baseline_table: &'a ArrowBaselineTable,
        record_batch: Arc<RecordBatch>,
    ) -> Result<ArrowCandidateTable<'a>, DriftArrowError> {
        let bl_schema = SchemaView::from_baseline_table(baseline_table);
        let candidate_schema = SchemaView::from_record_batch(&record_batch);
        if let SchemaValidationResult::Invalid(diff) =
            validate_schema(&bl_schema, &candidate_schema)
        {
            return Err(DriftArrowError::SchemaError(diff));
        }

        let mut table = HashMap::with_capacity(baseline_table.table.len());
        for (name, baseline_col) in &baseline_table.table {
            // SAFETY: schema validated above ensures column exists with matching type.
            let array = record_batch.column_by_name(name).unwrap().clone();
            table.insert(
                name.clone(),
                ArrowCandidateColumn::from_baseline_and_array(baseline_col, array)?,
            );
        }
        Ok(ArrowCandidateTable { table })
    }
}
